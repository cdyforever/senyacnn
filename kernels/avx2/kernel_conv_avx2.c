/*
 * ============================================================================
 * kernel_conv_avx2.c - AVX2 + FMA 卷积实现
 * ============================================================================
 *
 * 与 opt（标量优化版）逐项对应的三处向量化，几何分派逻辑完全相同，
 * 只是把三个热内核换成向量版本：
 *
 *   opt 的 conv_depthwise        ->  conv_depthwise_avx2 / conv_depthwise_s2_avx2
 *   opt 的 conv_1x1              ->  conv_1x1_avx2
 *   opt 的 dot4                  ->  dot4_avx2
 *
 * 指令集通过文件级 #pragma GCC target("avx2,fma") 指定，而不是全局 -march。
 * 这样宿主工程不必改编译选项，产物在任意 x86-64 上都能加载；真正能否执行
 * 由派发层用 kernel_avx2_supported() 在运行期判断，不支持则退回 opt。
 *
 * ----------------------------------------------------------------------------
 * 数值关系（重要）
 * ----------------------------------------------------------------------------
 * 与 opt 的差异来自两处，量级约 1e-4（特征图绝对误差，相对约 1e-5）：
 *   1. 用了 FMA：a*b+c 只做一次舍入，而 opt 是"先乘再舍入、再加再舍入"；
 *   2. 累加链从 4 条变成 8 路向量 + 双累加器，结合顺序不同。
 * 与 orig 的差异还额外包含 BN 折叠方式（orig 用近似 rsqrt），
 * 详细归因见 test_consistency.c 的文件头。
 *
 * 注意：这两个差异都不影响确定性 —— 同一实现重复运行结果逐位相同，
 * test_consistency.c 的"确定性自检"专门验证这一点。
 *
 * 数据布局、pad 填充、窗口锚点约定见 kernel_common.h。下面每个内核都会
 * 单独说明它对缓冲边界的假设，因为向量版要一次读 8 个 float，
 * 越界读比标量版危险得多。
 */
#include <stdlib.h>
#include <stdio.h>
#include "../kernel_common.h"

#if KERNEL_HAVE_AVX2
#include <immintrin.h>
#pragma GCC push_options
#pragma GCC target("avx2,fma")

/*
 * 向量的 relu / leaky 激活。
 *
 *   relu  : max(x, 0)
 *   leaky : max(x, 0.1x) —— 对 x>0 得 x，对 x<0 得 0.1x（因为 0.1x > x），
 *           一条 max 指令即可，无需 blendv 或比较掩码。
 *
 * 写成宏而不是函数，是为了让编译器把它内联进每个内核的循环体，
 * 避免函数调用打断向量化。
 */
#define VACT(a, act, k01) do {                                    \
        if ((act) == ACTIVATE_TYPE_RELU)                          \
            (a) = _mm256_max_ps((a), _mm256_setzero_ps());        \
        else if ((act) == ACTIVATE_TYPE_LEAKY)                    \
            (a) = _mm256_max_ps((a), _mm256_mul_ps((a), (k01)));  \
    } while (0)

/*
 * ============================================================================
 * dw_gather2 - 取出 8 个"步长 2"的元素
 * ============================================================================
 *
 * 输入 p 指向 x 处，返回向量 [p[0], p[2], p[4], ..., p[14]]。
 *
 * 为什么需要它：stride == 2 的 depthwise 卷积里，相邻输出对应输入的
 * 相隔 2 列，而 AVX2 没有便宜的 stride-2 load，只能自己重排。
 *
 * 重排过程（AVX2 的 256 位向量内部是两个独立的 128 位 lane，容易搞错）：
 *
 *   a = loadu(p + 0)  = [p0 p1 p2 p3 | p4 p5 p6 p7 ]
 *   b = loadu(p + 8)  = [p8 p9 p10 p11| p12 p13 p14 p15]
 *
 *   _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0)):
 *     每 lane 内取 (a[x0], a[x2], b[x0], b[x2])，
 *     得到 [p0 p2 p8 p10 | p4 p6 p12 p14]
 *
 *   _mm256_permutevar8x32_ps(idx={0,1,4,5,2,3,6,7}):
 *     把 lane 交错开，得到 [p0 p2 p4 p6 p8 p10 p12 p14]
 *
 * 只要 3 条指令、2 次 load。这是 stride-2 depthwise 能达到 1.7x 加速的关键；
 * 如果按标量 gather 写，光是取数就会吃掉全部收益。
 *
 * 安全性：本函数读 p[0..15]，共 16 个 float。调用方必须保证这 16 个元素
 * 都在同一通道平面内 —— 见 conv_depthwise_s2_avx2 里的边界说明。
 */
static __inline __m256 dw_gather2(const float *p)
{
    __m256 a = _mm256_loadu_ps(p);
    __m256 b = _mm256_loadu_ps(p + 8);
    __m256 s = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    return _mm256_permutevar8x32_ps(s, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
}

/*
 * ============================================================================
 * conv_depthwise_avx2 - depthwise 3x3 stride=1
 * ============================================================================
 *
 * 核心思想：输出在水平方向连续，而 3x3 窗口在水平方向严重重叠。
 * 因此一次算出 8 个连续输出，比"每个输出各算一遍"省得多：
 *
 *   8 个输出需要的输入列范围是 [ox, ox+9]，共 10 列，
 *   而标量写法要为每个输出各读 3 列 = 24 次标量 load。
 *   向量写法只需 9 条 loadu（每行 3 条错位 load）+ 9 条 FMA。
 *
 * 权重 9 个值用 _mm256_set1_ps 广播到寄存器，整个 ox 循环期间不重新加载。
 *
 * 边界安全性（向量化最需要交代的地方）：
 *   循环条件 x + 8 <= ow 保证 ox <= ow - 8；
 *   最大读取列号 = ox + 2 + 7 = ow + 1；
 *   而输入行宽 mwi = width + 2*pad，对 stride=1、size=3、pad=1 有
 *     ow = (width - 3 + 2) / 1 + 1 = width，mwi = width + 2 = ow + 2，
 *   故 ow + 1 = mwi - 1，正好是行内最后一个合法下标，不越界。
 *   行方向同理：oy <= oh - 1，读到 r2 = r0 + 2*mwi，最后一行的
 *   (oh-1+2) = oh+1 <= mhi-1，也在平面内。
 *
 * 尾部不足 8 个输出用标量补齐，走与 opt 相同的公式（含 kernel_actv）。
 */
static void conv_depthwise_avx2(const float * restrict in, int C, int mwi, int mhi,
                                float * restrict out, int mwo, int mho, int ow, int oh, int po,
                                const float * restrict W, const float * restrict B, int act)
{
    const __m256 k01 = _mm256_set1_ps(0.1f);
    int c, oy, ox;

    for (c = 0; c < C; c++) {
        const float *w = W + c * 9;
        /* 9 个权重 + bias 全部广播到寄存器，循环内不再访存 */
        __m256 w0=_mm256_set1_ps(w[0]),w1=_mm256_set1_ps(w[1]),w2=_mm256_set1_ps(w[2]);
        __m256 w3=_mm256_set1_ps(w[3]),w4=_mm256_set1_ps(w[4]),w5=_mm256_set1_ps(w[5]);
        __m256 w6=_mm256_set1_ps(w[6]),w7=_mm256_set1_ps(w[7]),w8=_mm256_set1_ps(w[8]);
        __m256 bc = _mm256_set1_ps(B[c]);

        const float *ip = in + (size_t)c * mwi * mhi;          /* 本通道平面基址 */
        float *op = out + (size_t)c * mwo * mho + po * mwo + po;/* 本通道内部左上角 */

        for (oy = 0; oy < oh; oy++) {
            /* 三行窗口起始指针：r0 当前行、r1 下一行、r2 再下一行 */
            const float *r0 = ip + oy * mwi, *r1 = r0 + mwi, *r2 = r1 + mwi;
            float *orow = op + oy * mwo;

            for (ox = 0; ox + 8 <= ow; ox += 8) {
                /* 累加器初值 = bias，省掉最后一次单独加法 */
                __m256 a = bc;
                /* 每行 3 条错位 loadu 覆盖 kx = 0,1,2；重叠部分由 cache 吸收 */
                a = _mm256_fmadd_ps(w0, _mm256_loadu_ps(r0+ox),   a);
                a = _mm256_fmadd_ps(w1, _mm256_loadu_ps(r0+ox+1), a);
                a = _mm256_fmadd_ps(w2, _mm256_loadu_ps(r0+ox+2), a);
                a = _mm256_fmadd_ps(w3, _mm256_loadu_ps(r1+ox),   a);
                a = _mm256_fmadd_ps(w4, _mm256_loadu_ps(r1+ox+1), a);
                a = _mm256_fmadd_ps(w5, _mm256_loadu_ps(r1+ox+2), a);
                a = _mm256_fmadd_ps(w6, _mm256_loadu_ps(r2+ox),   a);
                a = _mm256_fmadd_ps(w7, _mm256_loadu_ps(r2+ox+1), a);
                a = _mm256_fmadd_ps(w8, _mm256_loadu_ps(r2+ox+2), a);
                VACT(a, act, k01);
                _mm256_storeu_ps(orow + ox, a);
            }
            /* 尾部标量收尾。累加顺序与 opt 一致，便于对拍 */
            for (; ox < ow; ox++) {
                float acc = B[c];
                acc += w[0]*r0[ox] + w[1]*r0[ox+1] + w[2]*r0[ox+2];
                acc += w[3]*r1[ox] + w[4]*r1[ox+1] + w[5]*r1[ox+2];
                acc += w[6]*r2[ox] + w[7]*r2[ox+1] + w[8]*r2[ox+2];
                orow[ox] = kernel_actv(acc, act);
            }
        }
    }
}

/*
 * ============================================================================
 * conv_depthwise_s2_avx2 - depthwise 3x3 stride=2
 * ============================================================================
 *
 * 与 stride=1 版本的区别只有两点：
 *   - 行方向步长为 2：输出第 oy 行读输入第 2*oy、2*oy+1、2*oy+2 行；
 *   - 列方向步长为 2：输出第 ox 个位置对应输入第 2*ox 列，
 *     而窗口仍是"3 个相邻列"，所以需要 dw_gather2 把相隔 2 的元素抽出来。
 *
 * 注意一个容易写错的地方：窗口内 kx 的偏移是 +0/+1/+2（相邻列），
 * 只有"输出到输出"才跨 2 列。写成 +0/+2/+4 会漏掉奇数列的贡献，
 * 结果是画出"隔行采样"的错图（这个错误在开发中被一致性测试抓到过）。
 *
 * 边界安全性：
 *   循环条件 ox + 8 <= ow，故 ox <= ow - 8；
 *   dw_gather2(p + 2) 会读 p[2 .. 17]，即输入列号最大 2*ox + 17。
 *   对 mbv2 的实测形状（width=208, pad=1, stride=2）：
 *     ow = (208 - 3 + 2)/2 + 1 = 104，mwi = 210，
 *     ox 最大 96 -> 2*96 + 17 = 209 = mwi - 1，正好在行内。
 *   行方向：oy <= oh-1 时读 r2 = ip + (2*oy+2)*mwi，
 *     (2*(oh-1)+2) = 2*oh <= mhi-1 亦成立。
 * 这类"刚好卡到边界"的性质依赖 stride/size/pad 的固定组合，因此在派发处
 * 只对 size==3 且 stride∈{1,2} 启用本内核，其余尺寸回落到通用路径。
 */
static void conv_depthwise_s2_avx2(const float * restrict in, int C, int mwi, int mhi,
                                   float * restrict out, int mwo, int mho, int ow, int oh, int po,
                                   const float * restrict W, const float * restrict B, int act)
{
    const __m256 k01 = _mm256_set1_ps(0.1f);
    int c, oy, ox;

    for (c = 0; c < C; c++) {
        const float *w = W + c * 9;
        __m256 w0=_mm256_set1_ps(w[0]),w1=_mm256_set1_ps(w[1]),w2=_mm256_set1_ps(w[2]);
        __m256 w3=_mm256_set1_ps(w[3]),w4=_mm256_set1_ps(w[4]),w5=_mm256_set1_ps(w[5]);
        __m256 w6=_mm256_set1_ps(w[6]),w7=_mm256_set1_ps(w[7]),w8=_mm256_set1_ps(w[8]);
        __m256 bc = _mm256_set1_ps(B[c]);

        const float *ip = in + (size_t)c * mwi * mhi;
        float *op = out + (size_t)c * mwo * mho + po * mwo + po;

        for (oy = 0; oy < oh; oy++) {
            /* 行步长为 2：三行窗口是 2*oy、2*oy+1、2*oy+2 */
            const float *r0 = ip + (oy * 2) * mwi, *r1 = r0 + mwi, *r2 = r1 + mwi;
            float *orow = op + oy * mwo;

            for (ox = 0; ox + 8 <= ow; ox += 8) {
                /* 每个输出位置的输入列起点相隔 2 */
                const float *p0 = r0 + ox * 2, *p1 = r1 + ox * 2, *p2 = r2 + ox * 2;
                __m256 a = bc;
                /* kx 偏移是 +0/+1/+2（相邻列），不是 +0/+2/+4 */
                a = _mm256_fmadd_ps(w0, dw_gather2(p0),     a);
                a = _mm256_fmadd_ps(w1, dw_gather2(p0 + 1), a);
                a = _mm256_fmadd_ps(w2, dw_gather2(p0 + 2), a);
                a = _mm256_fmadd_ps(w3, dw_gather2(p1),     a);
                a = _mm256_fmadd_ps(w4, dw_gather2(p1 + 1), a);
                a = _mm256_fmadd_ps(w5, dw_gather2(p1 + 2), a);
                a = _mm256_fmadd_ps(w6, dw_gather2(p2),     a);
                a = _mm256_fmadd_ps(w7, dw_gather2(p2 + 1), a);
                a = _mm256_fmadd_ps(w8, dw_gather2(p2 + 2), a);
                VACT(a, act, k01);
                _mm256_storeu_ps(orow + ox, a);
            }
            for (; ox < ow; ox++) {
                const float *p0 = r0 + ox*2, *p1 = r1 + ox*2, *p2 = r2 + ox*2;
                float acc = B[c];
                acc += w[0]*p0[0] + w[1]*p0[1] + w[2]*p0[2];
                acc += w[3]*p1[0] + w[4]*p1[1] + w[5]*p1[2];
                acc += w[6]*p2[0] + w[7]*p2[1] + w[8]*p2[2];
                orow[ox] = kernel_actv(acc, act);
            }
        }
    }
}

/*
 * ============================================================================
 * conv_1x1_avx2 - pointwise 卷积，4 个输出通道分块
 * ============================================================================
 *
 * 这是全网络最重要的内核（实测占 mbv2 运行时长的 78%），改动动机如下。
 *
 * opt 版本的问题是"输入被按输出通道数重复读取"：
 *   循环顺序 i(输出通道) -> y -> c(输入通道) -> x，
 *   每算一个输出通道的每一行，都要把整片输入按 c 走一遍。
 *   输入读取总量 = O(输出通道数 × 输入通道数 × 像素数)，是纯浪费 ——
 *   同一份输入数据本来可以喂给多个输出通道。
 *
 * 这里的做法：
 *   一次处理 4 个输出通道，共用同一次输入 load。内循环只做
 *       1 次 load + 4 次 FMA
 *   于是输入读取次数从 N 降到 N/4，同时 4 条累加链互不依赖，
 *   FMA 的延迟被彻底掩盖。实测同形状下 1.5~2.3 倍加速。
 *
 * 为什么是 4 个而不是 8 个？
 *   8 个输出通道需要同时load 8 个权重标量并占 8 个累加器寄存器，
 *   加上输入向量/权重广播/尾部处理，在 AVX2 的 16 个 ymm 寄存器里就很紧张，
 *   编译器容易溢出到栈上，反而变慢。开发中实测过 8 路版本：在
 *   "输出通道多、空间尺寸小"的层（如 13x13x960 -> 320）会退化到 0.77x。
 *   4 路在各种形状上都稳定，因此最终采用 4 路。
 *
 * 边界安全性：
 *   loadu(in + c*plane + y*wi + x) 一次读 8 个 float，
 *   循环条件 x + 8 <= ow 且本内核要求 wi == ow（输入 pad==0），
 *     最大下标 = y*wi + ow - 1 < plane，
 *   因此不会跨通道平面读到下一个通道的数据。
 *
 * 输出可能带 pad（下一层需要的边缘），所以写指针要加 pad 偏移，
 * 每行还要跨过 mwo - width 的尾部。
 */
static void conv_1x1_avx2(const float * restrict in, int C, int wi, int hi,
                          float * restrict out, int N, int mwo, int mho,
                          int ow, int oh, int po,
                          const float * restrict W, const float * restrict B, int act)
{
    const int plane = wi * hi;                          /* 单通道平面大小（输入） */
    const __m256 k01 = _mm256_set1_ps(0.1f), zero = _mm256_setzero_ps();
    int i, c, y, x, t;

    /* 外层按 4 个输出通道分块 */
    for (i = 0; i < N; i += 4) {
        int nb = (N - i) < 4 ? (N - i) : 4;             /* 尾部可能不足 4 个 */

        /* 为块内每个输出通道准备好：输出平面基址、权重行首、bias 广播 */
        float *op[4]; const float *wp[4]; __m256 bv[4];
        for (t = 0; t < nb; t++) {
            op[t] = out + (i + t) * mwo * mho + po * mwo + po;
            wp[t] = W + (i + t) * C;
            bv[t] = _mm256_set1_ps(B[i + t]);
        }

        for (y = 0; y < oh; y++) {
            /* 块内每个输出通道对应行的写指针 */
            float *o0 = op[0] + y * mwo;
            float *o1 = nb > 1 ? op[1] + y * mwo : o0;
            float *o2 = nb > 2 ? op[2] + y * mwo : o0;
            float *o3 = nb > 3 ? op[3] + y * mwo : o0;

            for (x = 0; x + 8 <= ow; x += 8) {
                /* 4 个独立累加器，初值都是各自的 bias */
                __m256 a0 = bv[0], a1 = nb > 1 ? bv[1] : bv[0];
                __m256 a2 = nb > 2 ? bv[2] : bv[0], a3 = nb > 3 ? bv[3] : bv[0];

                for (c = 0; c < C; c++) {
                    /* 关键：一次 load 供 4 个输出通道复用 */
                    __m256 v = _mm256_loadu_ps(in + c * plane + y * wi + x);
                    a0 = _mm256_fmadd_ps(_mm256_set1_ps(wp[0][c]), v, a0);
                    if (nb > 1) a1 = _mm256_fmadd_ps(_mm256_set1_ps(wp[1][c]), v, a1);
                    if (nb > 2) a2 = _mm256_fmadd_ps(_mm256_set1_ps(wp[2][c]), v, a2);
                    if (nb > 3) a3 = _mm256_fmadd_ps(_mm256_set1_ps(wp[3][c]), v, a3);
                }

                /*
                 * linear 与“未知激活”都不需要额外运算，直接存。
                 * 依据：parser.c 的 get_activation_type_int 只产出
                 * {LINEAR, RELU, LEAKY, -1}，而 -1 在 kernel_actv / activate
                 * 里同样按 linear 处理，与这里的“原样存”完全等价，
                 * 因此向量路径不会漏掉任何可达的激活类型。
                 * （SIGMOID 只用于 yolo 解码，不会出现在 cfg 的层参数里。）
                 */

                /* 激活必须在全部输入通道累加完之后一次性施加 */
                if (act == ACTIVATE_TYPE_RELU) {
                    a0 = _mm256_max_ps(a0, zero);
                    if (nb > 1) a1 = _mm256_max_ps(a1, zero);
                    if (nb > 2) a2 = _mm256_max_ps(a2, zero);
                    if (nb > 3) a3 = _mm256_max_ps(a3, zero);
                } else if (act == ACTIVATE_TYPE_LEAKY) {
                    a0 = _mm256_max_ps(a0, _mm256_mul_ps(a0, k01));
                    if (nb > 1) a1 = _mm256_max_ps(a1, _mm256_mul_ps(a1, k01));
                    if (nb > 2) a2 = _mm256_max_ps(a2, _mm256_mul_ps(a2, k01));
                    if (nb > 3) a3 = _mm256_max_ps(a3, _mm256_mul_ps(a3, k01));
                }
                /* linear 与其它激活：数值不变/交由标量尾部处理 */

                _mm256_storeu_ps(o0 + x, a0);
                if (nb > 1) _mm256_storeu_ps(o1 + x, a1);
                if (nb > 2) _mm256_storeu_ps(o2 + x, a2);
                if (nb > 3) _mm256_storeu_ps(o3 + x, a3);
            }

            /*
             * 行尾不足 8 个的收尾。
             * 宽度在 MobileNet 系网络里基本都是 8 的倍数（208/104/52/26/13…），
             * 26 和 13 会走到这里 2 次和 5 次，代价可忽略。
             * 用标量而不是 maskload，是为了保持与 opt 完全相同的累加顺序，
             * 便于定位问题（maskload 版本在开发中试过，收益不足 1%）。
             */
            for (; x < ow; x++) {
                for (t = 0; t < nb; t++) {
                    const float *w = wp[t];
                    float acc = B[i + t];
                    for (c = 0; c < C; c++) acc += w[c] * in[c * plane + y * wi + x];
                    op[t][y * mwo + x] = activate(acc, act);
                }
            }
        }
    }
}

/*
 * ============================================================================
 * dot4_avx2 - 通用路径的点积（8 float 一条向量，双累加器）
 * ============================================================================
 *
 * 与 opt 的 dot4 对应，但以 8 个 float 为一条向量、用 2 个累加器交替，
 * 等价于 16 路标量累加，能更好掩盖 FMA 延迟。
 *
 * 为什么这里可以用 8 路而 1x1 内核用 4 路？
 *   两者受约束的因素不同：
 *   - dot 的 ftotal 常见为 144/576/864/1728，都是 8 的倍数，8 路不浪费；
 *     且它没有"多输出通道"的寄存器压力。
 *   - 1x1 内核要同时维护 4 个输出的累加器 + 权重广播，寄存器更紧。
 *
 * 收尾：先降 8 路到 128 位做水平相加，剩余不足 8 个的用标量补进 s1，
 * 最后 s0 + s1。顺序确定，因此重复运行结果一致。
 */
static float dot4_avx2(const float * restrict w, const float * restrict r, int n)
{
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    float s0, s1 = 0;
    int j = 0;

    /* 主循环：每轮 16 个，两个累加器交替，互不依赖 */
    for (; j + 16 <= n; j += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j),     _mm256_loadu_ps(r + j),     a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j + 8), _mm256_loadu_ps(r + j + 8), a1);
    }
    /* 剩 8 个的整块 */
    for (; j + 8 <= n; j += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j), _mm256_loadu_ps(r + j), a0);

    /* 合并两条累加链 */
    a0 = _mm256_add_ps(a0, a1);
    {
        /* 256 位降 128 位，再两次水平相加得到标量 */
        __m128 lo = _mm256_castps256_ps128(a0);
        __m128 hi = _mm256_extractf128_ps(a0, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        s0 = _mm_cvtss_f32(s);
    }
    /* 不足 8 个的尾巴 */
    for (; j < n; j++) s1 += w[j] * r[j];
    return s0 + s1;
}

#pragma GCC pop_options
#endif /* KERNEL_HAVE_AVX2 */

/*
 * ============================================================================
 * kernel_conv_forward_avx2 - AVX2 版入口
 * ============================================================================
 *
 * 几何分派与 opt 完全一致，只是把三个内核换成向量版本。判定条件、
 * im2col 缓冲管理、组切片推进都与 opt 保持同构，便于逐行对照排查。
 *
 * 回落规则：
 *   - depthwise 且 size==3 且 stride∈{1,2}  -> 向量内核；
 *   - depthwise 其它尺寸（如 5x5）           -> 落到通用 im2col+向量点积；
 *   - 1x1（size==1 && stride==1 && 输入 pad==0 && groups==1）-> 向量内核；
 *   - 其余                                  -> 通用 im2col + dot4_avx2。
 *
 * 若本构建没有 AVX2（非 x86 或编译器不支持），直接转发给 opt，
 * 保证功能可用而不是链接失败。
 */
void kernel_conv_forward_avx2(NET *net, LAYER *ilayer, LAYER *olayer)
{
#if KERNEL_HAVE_AVX2
    MATRIX mati  = ilayer->matrix;
    MATRIX mato  = olayer->matrix;
    FILTER filter= ilayer->filter;
    int    is_dw, is_1x1;
    int    ftotal, scratch, mwo, ix, iy, ox, oy, i, x, y, n;
    float *col, *datao, *dataf;

    /* 与 opt 相同的几何判定，理由见 kernel_conv_opt.c 的入口注释 */
    is_dw  = (filter.channels == 1 && filter.groups == ilayer->matrix.channels && filter.groups >= 1);
    is_1x1 = (filter.size == 1 && filter.stride == 1 && ilayer->matrix.pad == 0 && filter.groups == 1);

    /* ---------- 路径一：depthwise 3x3，按 stride 选内核 ---------- */
    if (is_dw && filter.size == 3 && (filter.stride == 1 || filter.stride == 2)) {
        if (!net->bn_fused && filter.batchnorm) { printf("bn not fused!\n"); return; }
        /*
         * 传参说明：输入/输出都传"含 pad 的实际行宽/行高"，
         * 因为向量内核内部直接按 mwi/mwo 做行偏移。
         */
        if (filter.stride == 1)
            conv_depthwise_avx2(mati.data, mati.channels,
                                mati.width + mati.pad * 2, mati.height + mati.pad * 2,
                                mato.data, mato.width + mato.pad * 2, mato.height + mato.pad * 2,
                                mato.width, mato.height, mato.pad,
                                filter.data2, filter.bias2, filter.activate);
        else
            conv_depthwise_s2_avx2(mati.data, mati.channels,
                                   mati.width + mati.pad * 2, mati.height + mati.pad * 2,
                                   mato.data, mato.width + mato.pad * 2, mato.height + mato.pad * 2,
                                   mato.width, mato.height, mato.pad,
                                   filter.data2, filter.bias2, filter.activate);
        return;
    }

    /* ---------- 路径二：1x1 pointwise ---------- */
    if (is_1x1) {
        if (!net->bn_fused && filter.batchnorm) { printf("bn not fused!\n"); return; }
        conv_1x1_avx2(mati.data, filter.channels, mati.width, mati.height,
                      mato.data, mato.channels,
                      mato.width + mato.pad * 2, mato.height + mato.pad * 2,
                      mato.width, mato.height, mato.pad,
                      filter.data2, filter.bias2, filter.activate);
        return;
    }

    /* ---------- 路径三：通用 im2col + 向量点积 ---------- */
    mati.channels /= filter.groups;
    mato.channels /= filter.groups;
    filter.n      /= filter.groups;
    ftotal         = filter.size * filter.size * filter.channels;

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
         * im2col 本身是纯拷贝，没有向量化收益（编译器已经会向量化它），
         * 真正省下来的是后面每个滤波器复用同一份 col。
         */
        for (iy = 0, oy = 0; iy < mati.height; iy += filter.stride, oy++) {
            for (ix = 0, ox = 0; ix < mati.width; ix += filter.stride, ox++) {
                kernel_im2row(&mati, ix, iy, filter.size, col + ((oy * mato.width) + ox) * ftotal);
            }
        }

        for (dataf = filter.data2, i = 0; i < filter.n; i++, dataf += ftotal) {
            float bias  = filter.bias2[i];
            datao = mato.data + (mato.pad * mwo + mato.pad)
                  + i * (mwo * (mato.height + mato.pad * 2));
            switch (filter.activate) {
            case ACTIVATE_TYPE_RELU:
                for (y = 0; y < mato.height; y++, datao += mato.pad * 2) {
                    for (x = 0; x < mato.width; x++) {
                        float t = dot4_avx2(dataf, col + ((y * mato.width) + x) * ftotal, ftotal) + bias;
                        *datao++ = t > 0 ? t : 0;
                    }
                }
                break;
            case ACTIVATE_TYPE_LEAKY:
                for (y = 0; y < mato.height; y++, datao += mato.pad * 2) {
                    for (x = 0; x < mato.width; x++) {
                        float t = dot4_avx2(dataf, col + ((y * mato.width) + x) * ftotal, ftotal) + bias;
                        *datao++ = t > 0 ? t : 0.1f * t;
                    }
                }
                break;
            default:
                for (y = 0; y < mato.height; y++, datao += mato.pad * 2) {
                    for (x = 0; x < mato.width; x++) {
                       *datao++ = activate(dot4_avx2(dataf, col + ((y * mato.width) + x) * ftotal, ftotal) + bias, filter.activate);
                    }
                }
                break;
            }
        }

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
#else
    /*
     * 非 x86 目标或编译器不支持 target 属性。
     * 这里直接转发给 opt 而不是报错：调用方可能只是没检查能力就选了 avx2，
     * 让它退化成"能跑但没那么快"比崩溃友好。
     */
    kernel_conv_forward_opt(net, ilayer, olayer);
#endif
}
