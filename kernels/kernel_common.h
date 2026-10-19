/*
 * ============================================================================
 * kernel_common.h - 三套卷积内核共享的内部接口与不变量
 * ============================================================================
 *
 * 一、为什么要拆成三套
 * --------------------
 * 卷积是这类网络 95% 以上的耗时来源。本仓库保留三套实现，各自承担明确职责：
 *
 *   kernels/orig  : 原始标量实现。逐位可复现的历史基准，用于回归比对与排查。
 *   kernels/opt   : 优化标量实现。BN 折叠 + 按几何特化 + 多链点积，
 *                   不依赖任何 CPU 扩展，可移植性最好。
 *   kernels/avx2  : AVX2+FMA 向量化。在 opt 的基础上把三个热内核换成向量版本。
 *
 * 三者函数签名完全一致（NET*, LAYER*, LAYER*），因此 net_forward 可以按
 * net->conv_impl 直接切换，做正确性比对和性能 A/B。
 *
 * 二、数据布局（三套实现共用的唯一约定，改动它等于全盘重写）
 * ----------------------------------------------------------
 * 特征图为 CHW 平面存储：channel -> row -> col 依次连续。
 * 每个通道平面的**实际尺寸**是含 pad 边缘的：
 *
 *      mw = width  + 2 * pad
 *      mh = height + 2 * pad
 *      plane_size = mw * mh            （单位：float）
 *
 * 单个通道平面的内存形状（pad = 1 为例）：
 *
 *      +---------------------------+  <- data + 0
 *      | 0  0  0  0  0 ...         |  第 0 行全是 pad 区
 *      | 0  v  v  v  v ... 0       |  内部区域从 (pad, pad) 开始
 *      | 0  v  v  v  v ... 0       |  v = 真正的特征值
 *      | 0  0  0  0  0 ...         |  第 mh-1 行全是 pad 区
 *      +---------------------------+
 *
 * 内部像素 (x, y) 的地址 = data + y * mw + x，其中 data 已指向含 pad 的缓冲起点。
 * 也就是说 "内部坐标 = padded 坐标"，二者不差一个 pad 偏移。
 *
 * 三、pad 边缘的填充策略（性能关键，也是正确性关键）
 * --------------------------------------------------
 * 边缘值在 net_load 阶段由 matrix_fill_pad 填一次，之后跨帧有效：
 *
 *   - 普通层的输出边缘填 0；
 *   - maxpool 的输入边缘填 -FLT_MAX，使取 max 时自动忽略越界元素，
 *     从而让池化内核不必写任何边界判断分支。
 *
 * 推论（写内核时必须遵守）：
 *   所有内核**只写内部区域**，绝不触碰 pad 边缘。
 *   因为边缘值属于"下一层读取时的语义"，本层无权改写，否则跨帧会累积错值。
 *
 * 四、窗口锚点约定（曾在此踩坑，三套内核必须完全一致）
 * ---------------------------------------------------
 * 以 size x size 窗口为例，输出像素 (ox, oy) 的窗口覆盖
 *   padded[oy*stride .. oy*stride + size - 1][ox*stride .. ox*stride + size - 1]
 *
 * 注意锚点直接用 padded 坐标，**不加 +pad 修正**。因为 pad 已经包含在数组里，
 * 内部像素 (ix,iy) 的 padded 坐标恰好就是 (ix,iy)，所以这等价于
 * "以内部像素为中心的 size 窗口"，与 darknet 语义一致。
 *
 * 若误加 +pad，整个输出会平移一个像素，且在右/下边界越界读。历史版本在这里
 * 出过偏差，因此三套实现都严格遵守同一约定，并由 test_dw.c / test_1x1.c
 * 用独立的朴素实现逐元素校验。
 *
 * 五、数值约定
 * ------------
 * orig 使用 fast_inverse_sqrt 近似（约 0.2% 相对误差）做 BN 归一化，
 * 这是历史行为，必须保留以保证与旧基线逐位一致；
 * opt / avx2 改为加载时精确折叠（sqrtf），因此与 orig 存在系统性差异。
 *
 * 该差异的量级与归因方法见 test_consistency.c 的文件头说明：
 * 用 -DKERNEL_ORIG_EXACT_RSQRT 重编 orig 即可把这项隔离掉。
 *
 * 本文件只被 kernels 下各实现与派发代码包含，不属于公共接口。
 */
#ifndef _SENYACNN_KERNEL_COMMON_H_
#define _SENYACNN_KERNEL_COMMON_H_

#include "../internal.h"

/*
 * 本构建是否具备编译 AVX2 内核的条件。
 *
 * 判据只看「编译器是否支持函数级 target 属性」+「目标是否为 x86」，
 * **不看运行期 CPU**：这样同一份产物可以跨机器分发，能不能真正跑到
 * AVX2 由 kernel_avx2_supported() 在运行期决定，不支持就降级到 opt。
 *
 * 用函数级 #pragma GCC target 而不是全局 -march 的好处：
 *   1. 宿主工程的编译选项不必改，其它文件的代码生成不受影响；
 *   2. 不会因为 -march=native 把整个程序编译成只能在构建机上跑的指令集。
 */
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define KERNEL_HAVE_AVX2 1
#else
#define KERNEL_HAVE_AVX2 0
#endif

/* 卷积内核实现编号。数值顺序与目录 orig / opt / avx2 一一对应 */
enum {
    CONV_IMPL_ORIG = 0,   /* kernels/orig  ：原始标量，逐位可复现的基准 */
    CONV_IMPL_OPT  = 1,   /* kernels/opt   ：优化标量，BN 折叠 + 几何特化 */
    CONV_IMPL_AVX2 = 2    /* kernels/avx2  ：AVX2 + FMA 向量化 */
};

/*
 * 激活函数。linear 即恒等映射（直接返回 x）；sigmoid 只被 yolo 解码使用，
 * 卷积层里不会出现。未知 type 按 linear 处理，便于新增激活时向后兼容。
 */
float activate(float x, int type);

/*
 * 快速平方根倒数（Quake III 魔数 + 一次牛顿迭代），相对误差约 0.2%。
 *
 * 仅供 orig 路径使用：它在每个输出元素上现场调用一次来归一化 BN 方差，
 * 这是原始实现的性能短板之一（每元素一次近似开方）。
 * opt / avx2 改为加载时用精确 sqrtf 折叠进权重，内循环彻底没有 BN 运算。
 *
 * 保留它而不是直接换成 sqrtf 的原因：orig 要维持与历史基线逐位一致，
 * 而历史基线正是用这个近似算出来的。需要的精确版本可通过
 * KERNEL_ORIG_EXACT_RSQRT 编译开关切换（见 kernel_conv_orig.c）。
 */
float kernel_fast_inverse_sqrt(float x);

/*
 * 热循环用激活分派。
 *
 * 与上面的 activate() 区别在于：这里把最常用的 relu / leaky 展开成无分支表达式，
 * 编译器可以将其合并进向量内核（见 avx2 的 VACT 宏）；其余类型才转调通用函数。
 * 这样既避免在内层循环里付出一次函数调用，又保留了对新激活类型的兼容性。
 */
static __inline float kernel_actv(float x, int type)
{
    switch (type) {
    case ACTIVATE_TYPE_RELU : return x > 0 ? x : 0;
    case ACTIVATE_TYPE_LEAKY: return x > 0 ? x : 0.1f * x;
    default                 : return activate(x, type);
    }
}

/*
 * 大小写不敏感比较。只用于解析命令行里的内核名（orig/OPT/Avx2 都认），
 * 避免为了一个 strcasecmp 引入平台相关的头文件。
 */
static __inline int kernel_strcmpi(const char *a, const char *b)
{
    unsigned char ca, cb;
    while (*a && *b) {
        ca = (unsigned char)*a++;
        cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return (int)ca - (int)cb;
    }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

/*
 * im2row：把一个 size x size x channels 的邻域展平拷贝到 buf。
 *
 * 参数 x, y 是窗口左上角在 padded 数组中的坐标（见文件头"窗口锚点约定"，
 * 不加 +pad 修正）。输出排列为 channel-major、行内 fy -> fx，即
 *
 *   buf[ c * size*size + fy * size + fx ]
 *
 * 这个顺序与权重布局 [n][channels][fy][fx] 的内两维完全一致，
 * 因此后续点积可以直接按下标顺序展开，不需要任何重排。
 *
 * 循环里那两个指针回退是必要的：每拷贝完窗口的一小行，data 已经前进了 size，
 * 但下一小行要从同一通道平面的下一行行首开始，所以要减回去再跨一整行宽度。
 * 整个通道拷完后还要跨过一整个平面，才能到下一通道的同一窗口位置。
 */
void kernel_im2row(MATRIX *matrix, int x, int y, int fsize, float *buf);

/*
 * 三套卷积前向实现。签名与语义完全一致，只有实现策略不同，
 * 因此 net_forward 可以按 net->conv_impl 直接切换，便于逐位比对与性能 A/B。
 *
 *   kernel_conv_forward_orig : 原始标量。每个输出像素做一次 im2row，
 *                              单链累加 + 每元素重算 BN；对 yolo-fastest
 *                              保持与历史基线逐位一致。
 *   kernel_conv_forward_opt  : 优化标量。BN 折叠 + 按几何分派到
 *                              depthwise / 1x1 / 通用 im2col+dot4 三条路径。
 *   kernel_conv_forward_avx2 : 在 opt 基础上把三个内核换成 AVX2+FMA 版本。
 */
void kernel_conv_forward_orig(NET *net, LAYER *ilayer, LAYER *olayer);
void kernel_conv_forward_opt (NET *net, LAYER *ilayer, LAYER *olayer);
void kernel_conv_forward_avx2(NET *net, LAYER *ilayer, LAYER *olayer);

/*
 * 统一派发入口。按 net->conv_impl 选实现；avx2 在编译期缺失或运行期
 * CPU 不支持时自动降级到 opt（只提示一次，避免每帧刷屏）。
 * net_forward 只调用这一个函数，因此新增实现时不必改它。
 */
void kernel_conv_forward(NET *net, LAYER *ilayer, LAYER *olayer);

/*
 * 该实现是否需要 BN 折叠。
 * opt / avx2 只读 data2/bias2（折叠后的副本），因此必须先折叠；
 * orig 读原始 scale/mean/var，不需要折叠。返回 1 表示需要。
 */
int  kernel_impl_needs_bn_fold(int impl);

/* 内核名 -> CONV_IMPL_*；认不出的名字返回 -1，由调用方决定报错还是回退 */
int  kernel_impl_from_name(const char *name);

/* 本构建是否包含 AVX2 内核（编译期判定，与运行期 CPU 无关） */
int  kernel_avx2_compiled(void);

/* 运行期 CPU 是否支持 AVX2+FMA（结果缓存；不含编译期判断） */
int  kernel_avx2_supported(void);

#endif
