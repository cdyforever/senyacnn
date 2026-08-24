/*
 * net.c - 网络生命周期与调度：pad 填充 / 释放 / 输入预处理 / 前向调度 / 结构打印
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "internal.h"

/*
 * 把矩阵四周的 pad 边缘统一填成 val。
 * 仅在 net_load 阶段对每个输出缓冲调用一次：之后所有层只写内部区域，
 * 边缘值跨帧保持有效。maxpool 的输入边缘填 -FLT_MAX，使 max 自动忽略边缘。
 */
void matrix_fill_pad(MATRIX *mat, float val)
{
    float *data = mat->data;
    int    pad  = mat->pad;
    int    mw   = mat->width + pad * 2;
    int    mh   = mat->height+ pad * 2;
    int    i, x, y;
    for (i=0; i<mat->channels; i++) {
        for (y=0; y<pad; y++) {                       /* 上下各 pad 行整行填充 */
            for (x=0; x<mw ; x++) data[y * mw + x] = data[(mh - 1 - y) * mw + x] = val;
        }
        for (y=pad; y<mh-pad; y++) {                  /* 中间行的左右各 pad 列 */
            for (x=0; x<pad; x++) data[y * mw + x] = data[y * mw + (mw - 1 - x)] = val;
        }
        data += mw * mh;
    }
}

/*
 * 释放全部资源。所有权清单：
 *   slot0.data(输入缓冲/bbox 区)、layerbuf(所有输出 arena)、foldbuf(BN 折叠副本)、
 *   cnntempbuf(卷积临时缓冲)、weight_buf(原始权重)、net 本体(含层数组)。
 * 各层 matrix.data 除 slot0 外都指向 layerbuf 内部，不单独释放。
 */
void net_free(NET *net)
{
    if (!net) return;
    free(net->layer_list[0].matrix.data);
    free(net->layerbuf);
    free(net->foldbuf);
    free(net->cnntempbuf);
    free(net->weight_buf);
    free(net);
}

/*
 * 输入预处理：BGR 图像 -> letterbox 缩放写入网络输入平面(RGB)。
 * 缩放为最近邻；宽高比不一致时短边留白（缓冲已预清零），并记录
 * s1/s2 = 原边长/缩放边长 供 NMS 还原坐标。
 * 同时清空上一帧的检测框数组（bbox_list 与输入缓冲共用内存）。
 */
void net_input(NET *net, unsigned char *bgr, int w, int h, float *mean, float *norm)
{
    MATRIX *mat = NULL;
    int    linebytes, sw, sh, i, j;
    float  *p1, *p2, *p3;
    if (!net) return;

    memset(net->bbox_list, 0, sizeof(BBOX) * net->bbox_num); net->bbox_num = 0;
    mat = &(net->layer_list[0].matrix);
    if (mat->channels != 3) { printf("invalid input matrix channels: %d !\n", mat->channels); return; }

    /* letterbox：等比缩放到恰好填满网络输入的一个维度 */
    if (w * mat->height > h * mat->width) {
        sw = mat->width ; sh = mat->width * h / w;      /* 以宽度为准 */
        net->s1 = w; net->s2 = sw;
    } else {
        sh = mat->height; sw = mat->height* w / h;      /* 以高度为准 */
        net->s1 = h; net->s2 = sh;
    }
    linebytes = ((w * 3) + 3) & ~3; // align to 4 bytes
    /* R/G/B 三个平面的写指针（内部区域左上角） */
    p1 = mat->data + (mat->width + mat->pad * 2) * mat->pad + mat->pad;
    p2 = p1 + (mat->width + mat->pad * 2) * (mat->height + mat->pad * 2);
    p3 = p2 + (mat->width + mat->pad * 2) * (mat->height + mat->pad * 2);
    for (i=0; i<sh; i++) {
        for (j=0; j<sw; j++) {
            int x = j * net->s1 / net->s2, y = i * net->s1 / net->s2;
            /* BMP 行序为 BGR，这里拆面并转 RGB，再做 mean/norm 归一化 */
            *p1++ = (bgr[y * linebytes + x * 3 + 2] - mean[0]) * norm[0]; // r
            *p2++ = (bgr[y * linebytes + x * 3 + 1] - mean[1]) * norm[1]; // g
            *p3++ = (bgr[y * linebytes + x * 3 + 0] - mean[2]) * norm[2]; // b
        }
        p1 += (mat->width + mat->pad * 2) - j;          /* 行尾对齐到下一行行首 */
        p2 += (mat->width + mat->pad * 2) - j;
        p3 += (mat->width + mat->pad * 2) - j;
    }
}

/*
 * 前向推理主循环。
 * 所有输出缓冲已在 net_load 分配好，本函数无任何 malloc/free；
 * refcnt 统计仅供 net_dump 显示依赖关系。
 * 末尾调用 NMS 完成候选框筛选与坐标还原（阈值 0.5、min-IoU 模式）。
 */
void net_forward(NET *net)
{
    LAYER *ilayer, *olayer; int i, j;
    if (!net) return;
    /* 优化路径首次前向时折叠 BN（只做一次；原始路径跳过） */
    if (net->opt_conv && !net->bn_fused) net_bn_fold(net);
    /* 统计 route/shortcut 对各层的引用计数（dump 展示用） */
    for (i=0; i<net->layer_num; i++) {
        if (net->layer_list[i].depend_num > 0) {
            for (j=0; j<net->layer_list[i].depend_num; j++) {
                net->layer_list[net->layer_list[i].depend_list[j] + 1].refcnt++;
            }
        }
    }
    /* 主循环：ilayer 为当前层参数，olayer(ilayer+1) 描述其输出矩阵 */
    for (i=0; i<net->layer_num; i++) {
        ilayer = net->layer_list + i + 0;
        olayer = net->layer_list + i + 1;
        switch (ilayer->type) {
        case LAYER_TYPE_CONV    :
            if (net->opt_conv) layer_groupconv_forward_opt(net, ilayer, olayer);
            else               layer_groupconv_forward    (net, ilayer, olayer);
            break;
        case LAYER_TYPE_AVGPOOL : layer_avgmaxpool_forward(ilayer, olayer, 0);   break;
        case LAYER_TYPE_MAXPOOL : layer_avgmaxpool_forward(ilayer, olayer, 1);   break;
        case LAYER_TYPE_UPSAMPLE: layer_upsample_forward  (ilayer, olayer);      break;
        case LAYER_TYPE_DROPOUT : layer_dropout_forward   (ilayer, olayer);      break;
        case LAYER_TYPE_SHORTCUT: layer_shortcut_forward  (net, ilayer, olayer); break;
        case LAYER_TYPE_ROUTE   : layer_route_forward     (net, ilayer, olayer); break;
        case LAYER_TYPE_YOLO    : layer_yolo_forward      (net, ilayer);         break;
        }
    }
    net->bbox_num = nms(net->bbox_list, net->bbox_num, 0.5f, 1, net->s1, net->s2);
}

/* 打印逐层结构表：类型/核参数/输入输出尺寸/BN与激活/被引用次数 */
void net_dump(NET *net)
{
    int i, j;
    if (!net) return;
    printf("layer   type  filters fltsize  pad/strd input          output       bn/act  ref\n");
    for (i=0; i<net->layer_num; i++) {
        if (net->layer_list[i].type == LAYER_TYPE_YOLO) {
            printf("%3d %8s\n", i, get_layer_type_string(net->layer_list[i].type));
        } else if (net->layer_list[i].type == LAYER_TYPE_DROPOUT) {
            printf("%3d %8s %-38s -> %3dx%3dx%3d\n", i, get_layer_type_string(net->layer_list[i].type), "",
                net->layer_list[i+1].matrix.width, net->layer_list[i+1].matrix.height, net->layer_list[i+1].matrix.channels);
        } else if (net->layer_list[i].type == LAYER_TYPE_SHORTCUT || net->layer_list[i].type == LAYER_TYPE_ROUTE) {
            char strdeps[256] = "layers:", strnum[16];
            for (j=0; j<net->layer_list[i].depend_num; j++) {
                snprintf(strnum, sizeof(strnum), " %d", net->layer_list[i].depend_list[j]);
                strncat(strdeps, strnum, sizeof(strdeps) - 1);
            }
            printf("%3d %8s %-38s -> %3dx%3dx%3d           %d\n", i, get_layer_type_string(net->layer_list[i].type), strdeps,
                net->layer_list[i+1].matrix.width, net->layer_list[i+1].matrix.height, net->layer_list[i+1].matrix.channels, net->layer_list[i].refcnt);
        } else {
            printf("%3d %8s %3d/%3d %2dx%2dx%3d   %d/%2d   %3dx%3dx%3d -> %3dx%3dx%3d  %d/%-6s %d\n", i,
                get_layer_type_string(net->layer_list[i].type), net->layer_list[i].filter.n, net->layer_list[i].filter.groups,
                net->layer_list[i].filter.size, net->layer_list[i].filter.size, net->layer_list[i].filter.channels,
                net->layer_list[i].matrix.pad, net->layer_list[i].filter.stride,
                net->layer_list[i+0].matrix.width, net->layer_list[i+0].matrix.height, net->layer_list[i+0].matrix.channels,
                net->layer_list[i+1].matrix.width, net->layer_list[i+1].matrix.height, net->layer_list[i+1].matrix.channels,
                net->layer_list[i].filter.batchnorm, get_activation_type_string(net->layer_list[i].filter.activate), net->layer_list[i].refcnt);
        }
    }
    printf("total weights: %d, total bytes: %d\n", net->weight_size, (int)(sizeof(WEIGHTS_FILE_HEADER) + net->weight_size * sizeof(float)));
}
