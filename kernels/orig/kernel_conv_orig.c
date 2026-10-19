/*
 * ============================================================================
 * kernel_conv_orig.c - 原始卷积实现（基准路径，逐位可复现）
 * ============================================================================
 *
 * 这是本仓库最早的卷积实现，刻意保持原样不动。它的价值不是快，而是"确定性"：
 * 对 yolo-fastest 的输出与历史基线逐位一致，因此任何改动都能拿它做回归判据。
 * opt / avx2 都以本文件为语义参照，只允许浮点结合顺序上的差异。
 *
 * 算法（每个分组独立处理）
 * ------------------------
 *   for 每个输出位置 (ix, iy):
 *       im2row 把 size x size x channels 的窗口摊平成一行 ftotal 个数
 *       for 组内每个输出通道 i:
 *           acc = Σ_j W[i][j] * row[j]          <- 单链累加，顺序固定
 *           if 有 BN: acc = (acc - mean[i]) * rsqrt(var[i]+eps) * scale[i]
 *           out = activate(acc + bias[i])
 *
 * 三个已知的性能短板（也正是 opt / avx2 的出发点）
 * ------------------------------------------------
 *   1. BN 的 rsqrt 在每个输出元素上重算一次，而它只与输出通道有关，本可预先算好；
 *      更严重的是用的是 fast_inverse_sqrt 近似，每个元素一次近似开方。
 *   2. 单链浮点累加：acc 存在串行依赖，受 FP 加法延迟（约 4 cycle）限制，
 *      流水线跑不满。
 *   3. 输出写入按"整平面步长"跳跃（见下面 datao 的推进），cache 行利用率极低。
 *
 * 数值约定
 * --------
 * 为了与历史基线逐位一致，这里必须保留 fast_inverse_sqrt 这个近似。
 * 需要精确版本的场合（例如做内核一致性归因分析）可以定义
 * KERNEL_ORIG_EXACT_RSQRT，把近似换成 1/sqrtf()，此时 orig 与 opt/avx2
 * 的差异会从 1e-1 量级掉到 1e-4 量级 —— 见 test_consistency.c。
 *
 * 数据布局、pad 填充、窗口锚点约定见 kernel_common.h。
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../kernel_common.h"

/*
 * BN 归一化因子的取法。
 *
 * 默认走 kernel_fast_inverse_sqrt（历史行为，逐位可复现）；
 * 定义 KERNEL_ORIG_EXACT_RSQRT 后改用精确的 1/sqrtf。
 *
 * 为什么需要这个开关：orig 与 opt/avx2 之间 ~0.2% 的系统性差异全部来自这里
 * （opt/avx2 在加载时用精确 sqrtf 折叠 BN）。把它换成精确实现，就能验证
 * "opt/avx2 与 orig 数值等价、差异与内核实现无关"。这属于测试用开关，
 * 正常构建不要打开，否则 orig 就不再是历史基线了。
 */
#ifdef KERNEL_ORIG_EXACT_RSQRT
#define ORIG_INV_SQRT(x)  (1.0f / sqrtf(x))
#else
#define ORIG_INV_SQRT(x)  kernel_fast_inverse_sqrt(x)
#endif

void kernel_conv_forward_orig(NET *net, LAYER *ilayer, LAYER *olayer)
{
    /*
     * 用局部副本推进指针，避免改动 net 里的原始描述。
     * 约定：ilayer 描述本层输入与参数，olayer(ilayer+1) 描述本层输出。
     */
    MATRIX mati  = ilayer->matrix;
    MATRIX mato  = olayer->matrix;
    FILTER filter= ilayer->filter;
    int    ftotal, ix, iy, ox, oy, i, j, n;
    float *datao, *dataf;

    /*
     * 分组卷积：把"总通道数 / 总滤波器数"按 groups 拆成"每组的量"。
     * groups == 1 时这一步等于没变；depthwise 时 groups == channels，
     * 于是每组的 channels 和 n 都是 1。
     */
    mati.channels /= filter.groups;
    mato.channels /= filter.groups;
    filter.n      /= filter.groups;

    /* 一个输出像素对应的权重个数 = size * size * 每组输入通道数 */
    ftotal = filter.size * filter.size * filter.channels;

    /*
     * 临时缓冲只需容纳"一行"展开结果（本实现每次只展开一个窗口）。
     * 若已有缓冲够大就复用，否则重新分配 —— 这也是本实现的短板之一：
     * 它把 cnntempbuf 压到最小，但代价是 im2row 被调用 ftotal 次之多。
     * （opt 的通用路径反而要一整组 im2col 矩阵，因此 net_load 按最大需求分配。）
     */
    if (net->cnnbufsize < ftotal) {
        net->cnnbufsize = ftotal;
        if (net->cnntempbuf) free(net->cnntempbuf);
        net->cnntempbuf = malloc(net->cnnbufsize * sizeof(float));
        if (net->cnntempbuf == NULL) { printf("failed to allocate memory for cnntempbuf !"); return; }
    }

    for (n = 0; n < filter.groups; n++) {
        /*
         * 按 stride 遍历输入的内部区域。iy/ix 是 padded 坐标
         * （见 kernel_common.h 的窗口锚点约定，不加 +pad）。
         */
        for (iy = 0, oy = 0; iy < mati.height; iy += filter.stride, oy++) {
            for (ix = 0, ox = 0; ix < mati.width; ix += filter.stride, ox++) {

                /* 把以 (ix,iy) 为左上角的窗口摊平到 cnntempbuf，长度 ftotal */
                kernel_im2row(&mati, ix, iy, filter.size, net->cnntempbuf);

                /*
                 * 该输出像素在第 0 个输出通道平面上的地址。
                 * 注意 i 循环内 datao 会按整平面大小递增，因此这里只需算最前面那个，
                 * 后面靠 datao += plane 走 —— 正是这个跳跃导致 cache 不友好（短板 3）。
                 */
                datao = mato.data + (mato.pad + oy) * (mato.width + mato.pad * 2) + mato.pad + ox;

                for (dataf = filter.data, i = 0; i < filter.n; i++) {
                    /*
                     * 单链浮点累加。
                     * 故意不做 4 路展开：求和顺序固定才能与历史基线逐位一致。
                     * 这是"慢但确定"的取舍；要速度请用 opt / avx2。
                     */
                    for (*datao = 0, j = 0; j < ftotal; j++) *datao += *dataf++ * net->cnntempbuf[j];

                    /*
                     * 未折叠的 BN：out = ((dot - mean) * rsqrt(var + eps)) * scale
                     * eps = 1e-5 与 darknet 一致，避免 var 为 0 时除零。
                     * 注意这里每个元素都重算一次 rsqrt（短板 1）。
                     */
                    if (filter.batchnorm) {
                        *datao  = (*datao - filter.rolling_mean[i]) * ORIG_INV_SQRT(filter.rolling_variance[i] + 0.00001f);
                        *datao *= filter.scale[i];
                    }

                    /* 激活作用在 (累加值 + bias) 上 */
                   *datao  = activate(*datao + filter.bias[i], filter.activate);

                    /* 跳到下一个输出通道平面的同一像素位置（短路 3 的来源） */
                    datao += (mato.width + mato.pad * 2) * (mato.height + mato.pad * 2);
                }
            }
        }

        /*
         * 本组算完，把各指针推进到下一组的切片。
         * 注意 dataf 在 i 循环里已按 ftotal 前进过，这里直接按组大小重算，
         * 依赖"组内滤波器在权重缓冲中连续存放"这一布局。
         */
        mati.data  += (mati.width + mati.pad * 2) * (mati.height + mati.pad * 2) * mati.channels;
        mato.data  += (mato.width + mato.pad * 2) * (mato.height + mato.pad * 2) * mato.channels;
        filter.data+= filter.size * filter.size * filter.channels * filter.n;
        filter.bias            += filter.n;
        filter.scale           += filter.n;
        filter.rolling_mean    += filter.n;
        filter.rolling_variance+= filter.n;
    }
}
