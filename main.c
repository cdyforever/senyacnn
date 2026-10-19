/*
 * main.c - 演示入口
 *
 * 加载 BMP 图片与 darknet 格式的 .cfg/.weights 模型，执行推理并输出检测结果。
 *
 * 用法:
 *   ./test [bmp] [cfg] [weights]          orig 原始内核（逐位可复现的基准路径）
 *   ./test -opt  [bmp] [cfg] [weights]    opt  优化内核（BN 折叠 + dw/1x1 特化 + dot4）
 *   ./test -avx2 [bmp] [cfg] [weights]    avx2 AVX2+FMA 向量化内核
 *   ./test -impl=<name> ...               同上，显式指定 orig/opt/avx2
 *   ./test -n=<frames> ...                自定义计时帧数（默认 10）
 *
 * 三套内核放在 kernels/{orig,opt,avx2}，签名一致，可任意切换做正确性比对与性能 A/B：
 *   orig 对 yolo-fastest 的输出与历史基线逐位一致；
 *   opt / avx2 允许浮点误差，检测文本按容差比对；avx2 不可用时自动降级到 opt。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "internal.h"
#include "kernels/kernel_common.h"
#include "bmpfile.h"

#ifdef WIN32
#include <windows.h>
#define get_tick_count GetTickCount
#else
#include <time.h>
/* 单精度毫秒计时（单调时钟），用于统计推理耗时 */
static uint32_t get_tick_count()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
#endif

int main(int argc, char *argv[])
{
    /* 预处理参数：减均值再乘 norm。此处即简单的 /255 归一化到 [0,1] */
    static const float MEAN[3] = { 0.0f, 0.0f, 0.0f };
    static const float NORM[3] = { 1/255.f, 1/255.f, 1/255.f };
    char *file_bmp    = "test.bmp";
    char *file_cfg    = "yolo-fastest-1.1.cfg";
    char *file_weights= "yolo-fastest-1.1.weights";
    NET  *mynet       = NULL;
    BMP   mybmp       = {0};
    int   tick, i, impl = CONV_IMPL_ORIG, nframes = 10;
    char *impl_name = "orig";

    /*
     * 选择内核实现。兼容旧的 "-opt" 写法，另外提供 "-avx2" 与通用的 "-impl=名字"。
     * 选项可连续出现（如 -avx2 -n=20），处理时前移 argv，文件参数仍按位置解析。
     */
    while (argc > 1 && argv[1][0] == '-') {
        int parsed = -1;
        if      (0 == strcmp(argv[1], "-opt"))  parsed = CONV_IMPL_OPT;
        else if (0 == strcmp(argv[1], "-avx2")) parsed = CONV_IMPL_AVX2;
        else if (0 == strcmp(argv[1], "-orig")) parsed = CONV_IMPL_ORIG;
        else if (0 == strncmp(argv[1], "-impl=", 6)) parsed = kernel_impl_from_name(argv[1] + 6);
        else if (0 == strncmp(argv[1], "-n=", 3)) {
            nframes = atoi(argv[1] + 3);
            if (nframes < 1) nframes = 1;
            argc--; argv++;
            continue;
        }
        else break;                                  /* 不是选项，当作文件名 */
        if (parsed < 0) { printf("unknown impl: %s (use orig/opt/avx2)\n", argv[1]); return -1; }
        impl = parsed;
        argc--; argv++;
    }
    if (impl == CONV_IMPL_AVX2) impl_name = "avx2";
    else if (impl == CONV_IMPL_OPT) impl_name = "opt";

    /* 选项处理完，剩余参数按位置解释：图片 / cfg / weights */
    if (argc > 1) file_bmp    = argv[1];
    if (argc > 2) file_cfg    = argv[2];
    if (argc > 3) file_weights= argv[3];
    printf("file_bmp    : %s\n", file_bmp    );
    printf("file_cfg    : %s\n", file_cfg    );
    printf("file_weights: %s\n", file_weights);

    if (0 != bmp_load(&mybmp, file_bmp)) { printf("failed to load bmp file: %s !\n", file_bmp); return -1; }
    mynet = net_load(file_cfg, file_weights);
    /*
     * 在首次 net_forward 之前设定实现：net_set_conv_impl 会清 bn_fused，
     * 让下一次前向按新实现的需要决定是否折叠 BN。
     */
    if (mynet) {
        int eff;
        net_set_conv_impl(mynet, impl);
        eff = net_effective_conv_impl(mynet);
        printf("conv kernel : %s (requested %s)\n",
               eff == CONV_IMPL_AVX2 ? "avx2" : (eff == CONV_IMPL_OPT ? "opt" : "orig"), impl_name);
    }
    net_dump(mynet);

    /* 连续推理 nframes 帧取总耗时，摊平首帧的缓存冷启动影响 */
    tick = (int)get_tick_count();
    for (i=0; i<nframes; i++) {
        net_input  (mynet, mybmp.pdata, mybmp.width, mybmp.height, (float*)MEAN, (float*)NORM);
        net_forward(mynet);
    }
    printf("%dms\n", (int)get_tick_count() - (int)tick);

    /* 打印检测结果并在原图上画框（坐标已被 NMS 还原为原图尺度） */
    for (i=0; i<mynet->bbox_num; i++) {
        printf("score: %.2f, category: %2d, rect: (%3d %3d %3d %3d)\n", mynet->bbox_list[i].score, mynet->bbox_list[i].type,
            (int)mynet->bbox_list[i].x1, (int)mynet->bbox_list[i].y1, (int)mynet->bbox_list[i].x2, (int)mynet->bbox_list[i].y2);
        bmp_rectangle(&mybmp, (int)mynet->bbox_list[i].x1, (int)mynet->bbox_list[i].y1, (int)mynet->bbox_list[i].x2, (int)mynet->bbox_list[i].y2, 0, 255, 0);
    }
    net_free(mynet);
    bmp_save(&mybmp, "out.bmp");
    bmp_free(&mybmp);
    return 0;
}
