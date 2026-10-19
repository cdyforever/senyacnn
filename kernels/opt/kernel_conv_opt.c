/*
 * ============================================================================
 * kernel_conv_opt.c - 优化版卷积实现（纯标量，不依赖任何 CPU 扩展）
 * ============================================================================
 *
 * 相对 orig 的三项改进，也是 avx2 版本的对照基线：
 *
 *   1. BN 折叠。orig 在每个输出元素上算 (dot - mean) * rsqrt(var + eps) * scale，
 *      每个元素一次近似开方；这里改为在加载阶段一次性折叠进权重副本：
 *
 *          a_j  = scale_j / sqrt(var_j + eps)
 *          W'_jk = W_jk * a_j
 *          b'_j = bias_j - mean_j * a_j
 *
 *      于是内循环变成 out = activate(dot' + b')，彻底没有 BN 运算。
 *      折叠由 parser.c 的 net_bn_fold() 完成，结果放在 filter.data2 / bias2；
 *      原始 data / bias 保持不动，orig 仍可正常使用。
 *
 *   2. 按层几何分派专用内核。MobileNet 系网络里绝大多数卷积是 1x1 或
 *      depthwise，通用 im2col 对它们纯属浪费（im2col 的拷贝量与计算量相当）。
 *      因此按 (size, stride, groups, pad) 分三条路径，见 kernel_conv_forward_opt。
 *
 *   3. 通用路径的写顺序改为"按行连续"。orig 是"每个输出像素跳遍所有通道平面"，
 *      这里改成"每个滤波器连续写完整平面"，cache 行利用率高得多。
 *
 * 另外点积从单链改成 4 链（dot4），打破 FP 加法的串行依赖 —— 见 dot4 的注释。
 *
 * 与 orig 的数值关系
 * ------------------
 *   - depthwise / 1x1：累加顺序与定义一致（通道升序），差异仅来自 BN 折叠方式。
 *   - 通用路径：dot4 的求和顺序与单链不同，属正常浮点差异。
 * 实际量级约 1e-4（特征图绝对值），详见 test_consistency.c。
 *
 * 数据布局、pad 填充、窗口锚点约定见 kernel_common.h。
 */
#include <stdlib.h>
#include <stdio.h>
#include "../kernel_common.h"

/*
 * ============================================================================
 * dot4 - 4 路独立累加器的点积
 * ============================================================================
 *
 * 为什么不是单链？
 *   单链 acc += w[j]*r[j] 的每次迭代都依赖上一次的结果，而浮点加法有约 4 cycle
 *   的延迟。以现代 CPU 每周期能发起 2 条 FMA 计算，单链等于把吞吐压到 1/4。
 *   拆成 4 条互不相关的链后，4 次加法可以同时在流水线上推进，典型提速 1.5~2 倍。
 *
 * 为什么是 4 而不是 8？
 *   通用路径的 ftotal（= size*size*channels）常见取值是 9、27、144、576 等，
 *   4 路在 27 这种小尺寸上也能吃到 3 轮，尾部浪费少；8 路会让小 ftotal 大量退化
 *   到尾部循环。向量版（avx2 的 dot4_avx2）才用 8 路。
 *
 * 求和顺序与 orig 不同，因此两者不是逐位一致 —— 这是可接受的浮点差异。
 * 但结果是**确定性**的：固定为 ((a0+a1)+(a2+a3))，便于回归测试比对。
 */
static float dot4(const float * restrict w, const float * restrict r, int n)
{
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0; int j;
    for (j = 0; j + 4 <= n; j += 4) {
        a0 += w[j  ] * r[j  ];
        a1 += w[j+1] * r[j+1];
        a2 += w[j+2] * r[j+2];
        a3 += w[j+3] * r[j+3];
    }
    /* 尾部并入 a0：保持"确定性"，且不影响正常范围（ftotal 多为 4 的倍数） */
    for (; j < n; j++) a0 += w[j] * r[j];
    return (a0 + a1) + (a2 + a3);
}

/*
 * ============================================================================
 * conv_depthwise - depthwise 卷积专用内核
 * ============================================================================
 *
 * 适用条件（由分派处判定）：filter.channels == 1 且 groups == 输入通道数，
 * 即每个输出通道只与"自己那一个"输入通道做窗口卷积。
 *
 * 为什么单独写而不用通用路径？
 *   depthwise 的 rank-1 特性使 im2col 完全失去意义：im2col 拷贝量与计算量之比
 *   在普通卷积里是 1/size²（值得摊），在 depthwise 里等于 1（拷贝一次只用一次）。
 *   因此这里直接在含 pad 的输入缓冲上滑窗，零拷贝。
 *
 * 为什么不需要边界判断？
 *   输入四边的 pad 区在 net_load 时已填 0（见 kernel_common.h），
 *   所以窗口越过内部边界时读到的是 0，正是 zero-padding 的语义。
 *
 * 输出只写内部区域，pad 边缘属于下一层的语义，本层不得触碰。
 *
 * 已知性能上限：这是三套实现里算术强度最低的内核（每个输出仅 size² 次乘加，
 * 却有 size² 次 load），因此吞吐天生低于 1x1 路径。avx2 版靠"一次 load 覆盖
 * 8 个输出"来改善，但仍有物理上限。
 */
static void conv_depthwise(MATRIX * restrict mati, MATRIX * restrict mato,
                           const float * restrict W, const float * restrict B,
                           int size, int stride, int act)
{
    int mwi = mati->width  + mati->pad * 2;   /* 输入缓冲行宽（含左右 pad） */
    int mhi = mati->height + mati->pad * 2;   /* 输入缓冲行数（含上下 pad） */
    int mwo = mato->width  + mato->pad * 2;   /* 输出缓冲行宽             */
    int mho = mato->height + mato->pad * 2;
    int c, ky, kx, ox, oy;

    for (c = 0; c < mati->channels; c++) {
        const float *w   = W + c * size * size;                                       /* 本通道的核       */
        float       *out = mato->data + c * mwo * mho + mato->pad * mwo + mato->pad;  /* 本通道内部左上角 */
        const float *in  = mati->data + c * mwi * mhi;                                /* 本通道平面基址   */

        for (oy = 0; oy < mato->height; oy++) {
            for (ox = 0; ox < mato->width; ox++) {
                /*
                 * 窗口左上角在 padded 坐标中的位置。
                 * 锚点不加 +pad（见 kernel_common.h 的窗口锚点约定）。
                 */
                const float *ip  = in + (oy * stride) * mwi + ox * stride;
                float        acc = 0;
                for (ky = 0; ky < size; ky++, ip += mwi) {
                    for (kx = 0; kx < size; kx++) acc += w[ky * size + kx] * ip[kx];
                }
                /* 激活在累加完之后施加，且用无分支版本 kernel_actv */
                out[oy * mwo + ox] = kernel_actv(acc + B[c], act);
            }
        }
    }
}

/*
 * ============================================================================
 * conv_1x1 - pointwise（1x1 stride 1）卷积专用内核
 * ============================================================================
 *
 * 这是 MobileNetV2 系网络的主要算力所在（实测占 mbv2 运行时长的 78%），
 * 因此它的写法直接决定整体性能。
 *
 * 为什么不用 im2col？
 *   1x1 的窗口就是一个点，im2col 会做"把每个像素拷到另一块内存"的纯搬运，
 *   然后对同一份数据做同样的点积 —— 白拷一遍。
 *
 * 这里的写法：把每个输出行先填成 bias，再按输入通道逐个"标量乘 + 行加"。
 *   优点：输出行常驻 L1，输入按行连续读，天然可向量化（avx2 版正是这么做的）。
 *   缺点：循环顺序是 i(输出通道) -> y(行) -> c(输入通道) -> x(列)，
 *         也就是**每算一行输出，都要把整片输入按输入通道扫一遍**，
 *         输入读取量被放大 O(输出通道数) 倍。
 *   这正是 avx2 版做"4 个输出通道共享一次输入 load"的动机 —— 见 kernel_conv_avx2.c。
 *
 * 前置条件（由分派处保证）：输入 pad == 0。因为本内核假设输入平面是紧致的
 * wi * height，若输入自带 pad 则 plane 计算会错。
 *
 * 输出可能带 pad（那是下一层作为输入时需要的边缘），所以写入时统一带 margin：
 * 起点是 pad*mwo + pad，每行还要跳过 mwo - width 个尾部元素。
 */
static void conv_1x1(MATRIX * restrict mati, MATRIX * restrict mato,
                     const float * restrict W, const float * restrict B, int act)
{
    int C     = mati->channels;             /* 输入通道数（分派处保证 groups==1） */
    int wi    = mati->width;                /* 输入行宽：本内核要求输入 pad==0     */
    int plane = wi * mati->height;          /* 单通道平面大小                      */
    int mwo   = mato->width + mato->pad * 2;
    int i, c, x, y;
    const float *in = mati->data;

    for (i = 0; i < mato->channels; i++) {
        const float *w   = W + i * C;       /* 第 i 个输出通道的 C 个系数 */
        float       *out = mato->data + i * (mwo * (mato->height + mato->pad * 2))
                         + mato->pad * mwo + mato->pad;

        /* 先在整片上累加，激活留到后面一次性做（避免每步都过一次激活） */
        for (y = 0; y < mato->height; y++) {
            float *orow = out + y * mwo;

            /* 行初始化为 bias：省掉最后一次单独的加法 */
            for (x = 0; x < mato->width; x++) orow[x] = B[i];

            for (c = 0; c < C; c++) {
                float        wc  = w[c];
                const float *src = in + c * plane + y * wi;    /* 输入第 c 通道的同一行 */
                /* 累加顺序固定为 c 升序，与数学定义一致，便于与 orig 对拍 */
                for (x = 0; x < mato->width; x++) orow[x] += wc * src[x];
            }
        }

        /*
         * 激活。这里把 relu / leaky 展开成独立循环，是因为它们足够常见，
         * 展开后编译器能把每个循环单独向量化；其余类型才退回通用 activate。
         */
        switch (act) {
        case ACTIVATE_TYPE_RELU:
            for (y = 0; y < mato->height; y++) {
                float *orow = out + y * mwo;
                for (x = 0; x < mato->width; x++) { float t = orow[x]; orow[x] = t > 0 ? t : 0; }
            }
            break;
        case ACTIVATE_TYPE_LEAKY:
            for (y = 0; y < mato->height; y++) {
                float *orow = out + y * mwo;
                for (x = 0; x < mato->width; x++) { float t = orow[x]; orow[x] = t > 0 ? t : 0.1f * t; }
            }
            break;
        default:
            for (y = 0; y < mato->height; y++) {
                float *orow = out + y * mwo;
                for (x = 0; x < mato->width; x++) orow[x] = activate(orow[x], act);
            }
            break;
        }
    }
}

/*
 * ============================================================================
 * kernel_conv_forward_opt - 优化版分组卷积入口（按几何分派）
 * ============================================================================
 *
 * 三条路径的判定条件与理由：
 *
 *   depthwise : groups == 输入通道数 且 每组输入通道数 == 1
 *               窗口卷积不适合 im2col（拷贝量 ≈ 计算量），走零拷贝直卷积。
 *
 *   1x1       : size==1 && stride==1 && 输入 pad==0 && groups==1
 *               窗口退化为单点，走"按行累加"，避免 im2col 的纯搬运。
 *               要求输入 pad==0，否则单通道平面不再是紧致的 wi*height。
 *
 *   其余      : 整组 im2col + 滤波器主序 GEMM
 *               通用路径。先把整组所有输出像素的窗口摊平成一个大矩阵
 *               （ftotal 行 × ow*oh 列），之后每个滤波器都复用这份只读数据，
 *               于是 im2col 的拷贝成本被"滤波器个数"摊薄。
 *
 * 三条路径都要求 BN 已折叠（bn_fused），因为它们只认 data2 / bias2。
 * 若调用方忘了折叠，这里会打印提示并直接返回，而不是算出一堆错值。
 */
void kernel_conv_forward_opt(NET *net, LAYER *ilayer, LAYER *olayer)
{
    MATRIX mati  = ilayer->matrix;
    MATRIX mato  = olayer->matrix;
    FILTER filter= ilayer->filter;
    int    is_dw, is_1x1;
    int    ftotal, scratch, mwo, ix, iy, ox, oy, i, x, y, n;
    float *col, *datao, *dataf;

    /*
     * 几何判定。
     * filter.channels 在 parser.c 里已算成"每组输入通道数"，
     * 所以 depthwise 的判据是它等于 1 且 groups 覆盖全部输入通道。
     */
    is_dw  = (filter.channels == 1 && filter.groups == ilayer->matrix.channels && filter.groups >= 1);
    is_1x1 = (filter.size == 1 && filter.stride == 1 && ilayer->matrix.pad == 0 && filter.groups == 1);

    /* ---------- 路径一：depthwise 直卷积 ---------- */
    if (is_dw) {
        if (!net->bn_fused && filter.batchnorm) { printf("bn not fused!\n"); return; }
        conv_depthwise(&mati, &mato, filter.data2, filter.bias2, filter.size, filter.stride, filter.activate);
        return;
    }

    /* ---------- 路径二：1x1 pointwise ---------- */
    if (is_1x1) {
        if (!net->bn_fused && filter.batchnorm) { printf("bn not fused!\n"); return; }
        conv_1x1(&mati, &mato, filter.data2, filter.bias2, filter.activate);
        return;
    }

    /* ---------- 路径三：通用 im2col + 滤波器主序 GEMM ---------- */
    mati.channels /= filter.groups;
    mato.channels /= filter.groups;
    filter.n      /= filter.groups;
    ftotal         = filter.size * filter.size * filter.channels;

    /*
     * 本路径要一整组 im2col 矩阵：ftotal 行 × (ow * oh) 列。
     * 这是全网络里最大的临时缓冲（mbv2 实测约 5.9 MB），net_load 已按
     * "所有卷积层的最大值"一次性分配好，这里只做兜底检查。
     */
    scratch = ftotal * mato.width * mato.height;
    if (net->cnnbufsize < scratch) {
        net->cnnbufsize = scratch;
        if (net->cnntempbuf) free(net->cnntempbuf);
        net->cnntempbuf = malloc(net->cnnbufsize * sizeof(float));
        if (net->cnntempbuf == NULL) { printf("failed to allocate memory for cnntempbuf !"); return; }
    }

    col = net->cnntempbuf;
    mwo = mato.width + mato.pad * 2;

    for (n = 0; n < filter.groups; n++) {
        /*
         * 步骤 1：im2col。把本组所有输出像素的窗口摊平。
         * 列索引 = oy * ow + ox，因此第 (y,x) 个输出像素的展开数据在
         * col + (y*ow + x) * ftotal 处，长度 ftotal。
         * 拷贝总量与 orig 相同（每输出像素一次），但这里只做一次，
         * 随后被组内 filter.n 个滤波器复用。
         */
        for (iy = 0, oy = 0; iy < mati.height; iy += filter.stride, oy++) {
            for (ix = 0, ox = 0; ix < mati.width; ix += filter.stride, ox++) {
                kernel_im2row(&mati, ix, iy, filter.size, col + ((oy * mato.width) + ox) * ftotal);
            }
        }

        /*
         * 步骤 2：滤波器主序 GEMM。
         * 外层是滤波器、内层是"行优先"的输出像素，因此写出的是连续地址
         * —— 这正是相对 orig 的第 3 项改进（orig 是整平面跳跃写）。
         *
         * datao 的行推进细节：每写完一行，要跳过输出平面右侧的 pad 列，
         * 所以 y 循环的步进是 datao += mato.pad * 2。
         */
        for (dataf = filter.data2, i = 0; i < filter.n; i++, dataf += ftotal) {
            float bias  = filter.bias2[i];    /* BN 折叠后的等效 bias */
            datao = mato.data + (mato.pad * mwo + mato.pad)
                  + i * (mwo * (mato.height + mato.pad * 2));

            /* 激活按层统一，因此在外面分支一次，避免内层每次判断 */
            switch (filter.activate) {
            case ACTIVATE_TYPE_RELU:
                for (y = 0; y < mato.height; y++, datao += mato.pad * 2) {
                    for (x = 0; x < mato.width; x++) {
                        float t = dot4(dataf, col + ((y * mato.width) + x) * ftotal, ftotal) + bias;
                        *datao++ = t > 0 ? t : 0;
                    }
                }
                break;
            case ACTIVATE_TYPE_LEAKY:
                for (y = 0; y < mato.height; y++, datao += mato.pad * 2) {
                    for (x = 0; x < mato.width; x++) {
                        float t = dot4(dataf, col + ((y * mato.width) + x) * ftotal, ftotal) + bias;
                        *datao++ = t > 0 ? t : 0.1f * t;
                    }
                }
                break;
            default:
                for (y = 0; y < mato.height; y++, datao += mato.pad * 2) {
                    for (x = 0; x < mato.width; x++) {
                       *datao++ = activate(dot4(dataf, col + ((y * mato.width) + x) * ftotal, ftotal) + bias, filter.activate);
                    }
                }
                break;
            }
        }

        /*
         * 推进到下一组切片。
         * data 与 data2 必须同步推进（data 供 orig 用，data2 供本路径用），
         * bias / bias2 / scale / mean / var 同理 —— 这五个指针是平行的参数数组，
         * 少推进一个就会让下一组读到上一组的参数，是非常隐蔽的错。
         */
        mati.data  += (mati.width + mati.pad * 2) * (mati.height + mati.pad * 2) * mati.channels;
        mato.data  += (mato.width + mato.pad * 2) * (mato.height + mato.pad * 2) * mato.channels;
        filter.data  += ftotal * filter.n;
        filter.data2 += ftotal * filter.n;
        filter.bias            += filter.n;
        filter.bias2           += filter.n;
        filter.scale           += filter.n;
        filter.rolling_mean    += filter.n;
        filter.rolling_variance+= filter.n;
    }
}
