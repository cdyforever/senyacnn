/*
 * ============================================================================
 * kernel_common.c - 三套卷积内核的运行期派发
 * ============================================================================
 *
 * 职责很窄：把 net_forward 的一次卷积调用，转发到 net->conv_impl 指定的那一套
 * 实现上，并在 AVX2 不可用时安全降级。共用数学在 kernel_shared.c。
 *
 * 为什么不把派发写进 net.c
 * ------------------------
 * net.c 关心的是"有哪些层、按什么顺序跑"，不该知道"卷积有几套实现、
 * 当前机器支持哪一套"。把选择逻辑收在这里，新增一套实现（比如 NEON）时
 * 只需要加一个 enum 值和一个 case，net.c 一行都不用改。
 *
 * 降级策略
 * --------
 * 选 avx2 但实际跑不了的情况有两种，都必须降级到 opt，否则要么编译不过、
 * 要么运行期非法指令崩溃：
 *
 *   1. 编译期没编入 AVX2 内核（非 x86 目标，或编译器不支持 target 属性）；
 *   2. 编入了，但当前 CPU 不支持 AVX2+FMA。
 *
 * 两种都只提示一次（static 标志），避免每层每帧刷屏 —— 一个 100 层的网络
 * 会调用本函数 100 次/帧。
 */
#include <stdio.h>
#include "../kernel_common.h"

/*
 * 内核名 -> CONV_IMPL_*。
 *
 * 认不出返回 -1，由调用方决定是报错还是回退：main.c 选择报错（用户显式写了
 * 一个不存在的名字，静默回退会让人困惑）；而测试程序可以据此跳过。
 * 比较用 kernel_strcmpi，因此 "-impl=AVX2" 与 "-impl=avx2" 等价。
 */
int kernel_impl_from_name(const char *name)
{
    if (!name) return -1;
    if (!kernel_strcmpi(name, "orig")) return CONV_IMPL_ORIG;
    if (!kernel_strcmpi(name, "opt"))  return CONV_IMPL_OPT;
    if (!kernel_strcmpi(name, "avx2")) return CONV_IMPL_AVX2;
    return -1;
}

/*
 * 统一派发入口。net_forward 对每个卷积层调用一次。
 *
 * 降级判断放在这里而不是在 net_load 里，是因为它依赖运行期 CPU 能力，
 * 而 net_load 应保持"纯解析+分配"的语义，不掺入行为决策。
 */
void kernel_conv_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    int impl = net->conv_impl;

    /* 情况 1：产物里没有 AVX2 内核 */
    if (impl == CONV_IMPL_AVX2 && !kernel_avx2_compiled()) {
        static int warned_compile = 0;
        if (!warned_compile) { printf("avx2 kernels not built, fallback to opt\n"); warned_compile = 1; }
        impl = CONV_IMPL_OPT;
    }
    /* 情况 2：有内核但 CPU 不支持（缺 AVX2 或缺 FMA 都会落到这里） */
    if (impl == CONV_IMPL_AVX2 && !kernel_avx2_supported()) {
        static int warned_cpu = 0;
        if (!warned_cpu) { printf("cpu lacks avx2/fma, fallback to opt\n"); warned_cpu = 1; }
        impl = CONV_IMPL_OPT;
    }

    switch (impl) {
    case CONV_IMPL_AVX2:
#if KERNEL_HAVE_AVX2
        kernel_conv_forward_avx2(net, ilayer, olayer);
        break;
#endif
        /* 编译期没有 AVX2 时，上面的 if 已把 impl 改成 OPT，走不到这里；
         * 这里不再写 fallthrough，让编译器的 -Wimplicit-fallthrough 保持干净。 */

    case CONV_IMPL_OPT:
        kernel_conv_forward_opt(net, ilayer, olayer);
        break;

    case CONV_IMPL_ORIG:
    default:
        /* 未知值按最保守的实现处理：orig 不依赖任何扩展，也不会读未折叠的 BN */
        kernel_conv_forward_orig(net, ilayer, olayer);
        break;
    }
}

/*
 * 该实现是否需要 BN 折叠。
 *
 * orig 现场用 scale / rolling_mean / rolling_variance 做归一化，不需要折叠；
 * opt / avx2 只读折叠后的 data2 / bias2，必须先折叠。
 *
 * 注意 avx2 降级到 opt 时仍然需要折叠（opt 也要 data2/bias2），
 * 因此这里对 avx2 一律返回 1，不受"是否真的跑得起来"影响 —— 反正两种归属
 * 都需要折叠，判断反而简单且不会错。
 */
int kernel_impl_needs_bn_fold(int impl)
{
    return impl != CONV_IMPL_ORIG;
}
