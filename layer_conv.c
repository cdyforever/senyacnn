/*
 * layer_conv.c - 卷积层入口（保留历史文件名，实际实现已拆分到 kernels/）
 *
 * 三套实现按目录分开放置，签名与语义一致，可运行期切换、逐位比对与性能 A/B：
 *
 *   kernels/orig/  kernel_conv_orig.c   原始标量实现（基准路径，逐位可复现）
 *   kernels/opt/   kernel_conv_opt.c    优化标量实现（BN 折叠 + 几何特化 + dot4）
 *   kernels/avx2/  kernel_conv_avx2.c   AVX2+FMA 向量化实现
 *   kernels/common/ kernel_common.c     共享件（activate / im2row）+ 派发逻辑
 *
 * 共享的坐标约定与特征图布局说明见 kernels/kernel_common.h。
 *
 * 本文件只保留两个历史符号的转发，避免既有调用点（net.c、测试）失效：
 *   layer_groupconv_forward     -> kernel_conv_forward_orig
 *   layer_groupconv_forward_opt -> kernel_conv_forward_opt
 */
#include "kernels/kernel_common.h"

/* 原始路径：与历史基线逐位一致，作为回归基准 */
void layer_groupconv_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    kernel_conv_forward_orig(net, ilayer, olayer);
}

/* 优化标量路径：BN 折叠 + depthwise/1x1 特化 + dot4 */
void layer_groupconv_forward_opt(NET *net, LAYER *ilayer, LAYER *olayer)
{
    kernel_conv_forward_opt(net, ilayer, olayer);
}
