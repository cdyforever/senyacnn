#!/bin/sh
# build-win.sh - MinGW/MSYS 构建（Windows 产物为 test.exe）
#
# 卷积内核按实现分目录，公共件与派发在 kernels/common：
#   kernels/orig   : 原始标量实现（基准，逐位可复现）
#   kernels/opt    : 优化标量实现（BN 折叠 + 几何特化 + dot4）
#   kernels/avx2   : AVX2 + FMA 向量化实现
#   kernels/common : kernel_shared.c（activate/im2row/CPU 检测）+ kernel_common.c（派发）
#
# AVX2 内核用文件内 #pragma GCC target 指定指令集，因此这里不需要加 -march，
# 产物在任意 x86-64 上都能加载（派发时会检测 CPU 能力并按需降级到 opt）。
gcc -Wall -O3 \
    bmpfile.c parser.c layer_conv.c layer_pool.c layer_misc.c layer_yolo.c nms.c net.c main.c \
    kernels/common/kernel_shared.c \
    kernels/common/kernel_common.c \
    kernels/orig/kernel_conv_orig.c \
    kernels/opt/kernel_conv_opt.c \
    kernels/avx2/kernel_conv_avx2.c \
    -o test.exe -lm
