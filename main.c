/*
 * main.c - 演示入口
 *
 * 加载 BMP 图片与 darknet 格式的 .cfg/.weights 模型，执行推理并输出检测结果。
 *
 * 用法:
 *   ./test [bmp] [cfg] [weights]          走原始内核（逐位可复现的基准路径）
 *   ./test -opt [bmp] [cfg] [weights]     走优化内核（BN 折叠 + depthwise/1x1 特化 + dot4）
 *
 * 两种路径并存，便于做正确性比对与性能 A/B 测试：
 *   原始路径对 yolo-fastest 模型的输出与历史基线逐位一致；
 *   优化路径允许浮点误差，检测文本按容差比对。
 */
#include <stdio.h>
#include <string.h>
#include "internal.h"
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
    int   tick, i, opt_flag = 0;

    /* "-opt" 只能作为第 1 个参数；整体前移 argv 使后续文件参数仍按位置解析 */
    if (argc > 1 && 0 == strcmp(argv[1], "-opt")) { opt_flag = 1; argc--; argv++; }
    if (argc > 1) file_bmp    = argv[1];
    if (argc > 2) file_cfg    = argv[2];
    if (argc > 3) file_weights= argv[3];
    printf("file_bmp    : %s\n", file_bmp    );
    printf("file_cfg    : %s\n", file_cfg    );
    printf("file_weights: %s\n", file_weights);

    if (0 != bmp_load(&mybmp, file_bmp)) { printf("failed to load bmp file: %s !\n", file_bmp); return -1; }
    mynet = net_load(file_cfg, file_weights);
    /* 必须在首次 net_forward 之前设置：优化路径首次前向会触发 BN 折叠 */
    if (mynet) mynet->opt_conv = opt_flag;
    net_dump(mynet);

    /* 连续推理 10 帧取总耗时，摊平首帧的缓存冷启动影响 */
    tick = (int)get_tick_count();
    for (i=0; i<10; i++) {
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
