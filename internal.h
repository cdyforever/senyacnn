/*
 * internal.h - 引擎内部共享声明
 *
 * 仅供引擎各 .c 文件包含，不属于公共 API。
 * 内容：编译器兼容宏、weights 文件头结构、激活类型枚举、跨文件的内部函数原型。
 */
#ifndef _SENYACNN_INTERNAL_H_
#define _SENYACNN_INTERNAL_H_

#ifdef _MSC_VER
/* MSVC 兼容：关闭 fopen 等 CRT 安全警告，补齐 snprintf 与定宽整型 */
#pragma warning(disable:4996)
#define snprintf _snprintf
typedef int                int32_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif

#include "senyacnn.h"

/*
 * darknet .weights 文件头：3 个版本号 + 1 个 uint64（训练时见过的图片数）。
 * 1 字节对齐保证跨编译器布局一致；读取时整体跳过即可。
 */
#pragma pack(1)
typedef struct {
    int32_t  ver_major, ver_minor, ver_revision;
    uint64_t net_seen;
} WEIGHTS_FILE_HEADER;
#pragma pack()

/* 激活类型。cfg 里只出现 linear/relu/leaky；sigmoid 供 yolo 解码内部使用 */
enum {
    ACTIVATE_TYPE_LINEAR ,
    ACTIVATE_TYPE_RELU   ,
    ACTIVATE_TYPE_LEAKY  ,
    ACTIVATE_TYPE_SIGMOID,
};

float activate(float x, int type);                    /* 定义于 layer_conv.c */
char* get_activation_type_string(int type);           /* 定义于 parser.c，dump 用 */
char* get_layer_type_string(int type);                /* 同上 */

void  matrix_fill_pad(MATRIX *mat, float val);        /* 定义于 net.c */
void  net_bn_fold(NET *net);                          /* 定义于 parser.c：BN 折叠进权重副本 */

/* 各层前向实现。ilayer 指向本层槽位，olayer = ilayer+1 即输出矩阵所在槽位 */
void layer_groupconv_forward     (NET *net, LAYER *ilayer, LAYER *olayer);
void layer_groupconv_forward_opt (NET *net, LAYER *ilayer, LAYER *olayer);
void layer_avgmaxpool_forward(LAYER *ilayer, LAYER *olayer, int flag);
void layer_upsample_forward  (LAYER *ilayer, LAYER *olayer);
void layer_dropout_forward   (LAYER *ilayer, LAYER *olayer);
void layer_shortcut_forward  (NET *net, LAYER *ilayer, LAYER *olayer);
void layer_route_forward     (NET *net, LAYER *ilayer, LAYER *olayer);
void layer_yolo_forward      (NET *net, LAYER *ilayer);

/* NMS 后处理，定义于 nms.c。min=1 时用交集/较小面积（常用于人脸），否则标准 IoU */
int nms(BBOX *bboxlist, int n, float threshold, int min, int s1, int s2);

#endif
