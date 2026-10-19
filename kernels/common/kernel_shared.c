/*
 * ============================================================================
 * kernel_shared.c - 三套卷积内核共用的基础件
 * ============================================================================
 *
 * 内容：激活函数、快速开方近似、im2row、CPU 能力检测。
 *
 * 为什么单独成一个编译单元
 * ------------------------
 * 这些函数被 orig / opt / avx2 三套实现共用，如果塞进任何一套里，另外两套就得
 * 复制一份，改一处忘一处必然出错。单独成文件还带来一个实际好处：单元测试
 * （test_dw / test_1x1）只链接本文件就能测单个内核，不必为凑齐符号而把整条
 * 派发链和三套实现都拉进来。
 *
 * 派发逻辑（kernel_conv_forward 等）放在 kernel_common.c，与本文件分开，
 * 就是为了让"共用数学"和"选哪套实现"两件事互不牵制。
 *
 * 数据布局、pad 填充策略、窗口锚点约定等全局性说明见 kernel_common.h。
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../kernel_common.h"

/*
 * 激活函数。linear 即恒等映射；sigmoid 只被 yolo 解码使用。
 *
 * 这里刻意不做无分支优化：它服务于"非热路径"和"未知激活类型"两条兜底路径。
 * 热循环请用 kernel_common.h 里的 kernel_actv()，那份会把 relu/leaky 展开，
 * 从而能被编译器内联并向量化。
 */
float activate(float x, int type)
{
    switch (type) {
    case ACTIVATE_TYPE_RELU   : return x > 0 ? x : 0;
    case ACTIVATE_TYPE_LEAKY  : return x > 0 ? x : 0.1f * x;
    case ACTIVATE_TYPE_SIGMOID: return 1.0f / (1.0f + (float)exp(-x));
    default: return x;
    }
}

/*
 * 快速平方根倒数：Quake III 的魔数初值 + 一次牛顿迭代，相对误差约 0.2%。
 *
 * 原理：
 *   1) 浮点数的位模式近似对应其对数，用 0x5F3759DF - (bits >> 1) 得到
 *      1/sqrt(x) 的一个廉价初值（这一步误差可达数个百分点）；
 *   2) 再用一次牛顿迭代 x = x * (1.5 - 0.5*y*x*x) 把误差压到 0.2% 量级。
 *
 * 为什么保留这个近似
 * ------------------
 * orig 路径在每个输出元素上调用它做 BN 归一化，历史基线就是这么算出来的。
 * 为了让 orig 继续与旧基线逐位一致（compare.sh 的回归判据），不能悄悄换成 sqrtf。
 *
 * 但它只是"历史正确"，并非"数值更好"：opt / avx2 在加载时用精确 sqrtf 折叠 BN，
 * 因此与 orig 有约 0.2% 的系统性差异 —— 这是 test_consistency.c 里
 * orig vs opt/avx2 大误差的全部来源，不是内核错误。
 * 需要时可用 -DKERNEL_ORIG_EXACT_RSQRT 把它换成精确实现来验证这一点。
 */
float kernel_fast_inverse_sqrt(float x)
{
    union { float f; int32_t i; } fui;
    float halfx = 0.5f * x;
    fui.f = x;                                  /* 按位读，为下面的魔数运算做准备 */
    fui.i = 0x5F3759DF - (fui.i >> 1);          /* 经典魔数初值                   */
    x     = fui.f;                              /* 按位写回，得到一个粗糙的估计值 */
    x     = x * (1.5f - halfx * x * x);         /* 一次牛顿迭代：收敛到约 0.2%    */
    return x;
}

/*
 * im2row：把以 padded 坐标 (x,y) 为左上角的 size x size x channels 邻域
 * 展平拷贝到 buf，排列为 channel-major、行内 fy -> fx，即
 *
 *     buf[ c * size*size + fy * size + fx ]
 *
 * 该顺序与权重布局 [n][channels][fy][fx] 的内两维严格一致，因此下游点积
 * 可以按数组下标顺序直接展开，无需任何重排（这是 im2col 方案能高效的前提）。
 *
 * 指针walk的细节（容易写错，注释在此）：
 *   每拷贝完窗口的一小行，data 已经前进 size 个位置，但下一小行要从本通道
 *   平面的"下一行行首"开始，所以先减回 size、再加一整行宽度 mw：
 *       data += mw - k        （此处 k == size）
 *   本通道的 size 行都拷完后，data 停在"最后一小行再往后一行"的位置，
 *   要先退回 size 行，再跨过整个平面，才能到达下一通道的同一窗口位置：
 *       data -= mw * size
 *       data += mw * mh
 *
 * 注意锚点不做 +pad 修正，理由见 kernel_common.h 的"窗口锚点约定"。
 * 这里不做任何越界检查：调用方保证 (x, y) 的窗口完全落在 padded 缓冲内
 * （由层的输出尺寸公式与 pad 填充策略共同保证）。
 */
void kernel_im2row(MATRIX *matrix, int x, int y, int fsize, float *buf)
{
    int mw = matrix->width + matrix->pad * 2;
    int mh = matrix->height + matrix->pad * 2;
    float *data = matrix->data + y * mw + x;
    int    i, j, k;

    for (i = 0; i < matrix->channels; i++) {
        for (j = 0; j < fsize; j++) {
            for (k = 0; k < fsize; k++) *buf++ = *data++;
            data += mw - k;                     /* 走到下一小行行首（k == fsize） */
        }
        data -= mw * fsize;                     /* 退回本通道窗口首行           */
        data += mw * mh;                        /* 跨过整个平面，到下一通道     */
    }
}

/*
 * 本构建是否包含 AVX2 内核。
 * 纯编译期判定：只要编译器支持函数级 target 属性且目标是 x86，就认为内核已编入。
 * 真正的"能不能跑"由下面的 kernel_avx2_supported() 在运行期回答。
 */
int kernel_avx2_compiled(void)
{
#if KERNEL_HAVE_AVX2
    return 1;
#else
    return 0;
#endif
}

/*
 * 运行期检测 CPU 是否支持 AVX2 + FMA。
 *
 * 两者都要：内核里用了 _mm256_fmadd_ps，只有 AVX2 没有 FMA 的机器上会触发
 * 非法指令，因此必须同时检查。
 *
 * 用 __builtin_cpu_supports 而不是手写 cpuid：
 *   - 它由编译器提供，不需要内联汇编，也不会与优化级别/目标属性冲突；
 *   - 调用前需要 __builtin_cpu_init()，GCC 在 main 之前通常已自动调用，
 *     但显式调用一次零成本且更稳妥。
 *
 * 结果缓存在 static 变量里：本函数会在每层的派发路径上被调用，
 * 重复做能力检测没有意义。cached = -1 表示"尚未检测"。
 */
int kernel_avx2_supported(void)
{
#if !KERNEL_HAVE_AVX2
    return 0;
#elif defined(__clang__) || defined(__GNUC__)
    #if defined(__x86_64__) || defined(__i386__)
        static int cached = -1;
        if (cached < 0) {
            __builtin_cpu_init();
            cached = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? 1 : 0;
        }
        return cached;
    #else
        return 0;                            /* 非 x86 目标：没有 AVX2 内核       */
    #endif
#else
    return 0;                                /* 编译器不支持该内建：保守当作没有 */
#endif
}
