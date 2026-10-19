/*
 * senyacnn.h - 轻量级 CNN 推理引擎公共 API
 *
 * 一个 darknet 风格的推理引擎：解析 darknet 的 .cfg 网络描述与 .weights 权重文件，
 * 支持层类型 conv / avgpool / maxpool / upsample / dropout / shortcut / route / yolo。
 * 全部缓冲在 net_load 时一次性分配（arena 方式），net_forward 内零堆操作。
 *
 * 数据布局约定：
 *   特征图为 CHW 平面存储（channel -> row -> col 依次连续）；
 *   每个通道平面实际占 (width + 2*pad) x (height + 2*pad)，四周为 pad 边缘区，
 *   边缘值在加载时填充一次（普通层填 0，maxpool 的输入边缘填 -FLT_MAX），
 *   之后所有写操作只触碰内部区域，因此跨帧有效。
 */
#ifndef _FFCNN_H_
#define _FFCNN_H_

/* 特征图/图像描述。data 指向含 pad 边缘的完整缓冲，见文件头注释的布局约定 */
typedef struct {
    int    width, height, channels, pad;
    float *data;
} MATRIX;

/*
 * 卷积核描述（仅 conv 层使用）。
 * 权重布局 [n][channels][size][size]，其中 channels 为输入通道数/groups；
 * batchnorm 参数按 scale / rolling_mean / rolling_variance 各 n 个 floats 存储；
 * data2/bias2 是 BN 折叠后的权重副本（opt 优化路径专用）：
 *   有 BN 时指向 foldbuf 中的 W' 与 b'，无 BN 时直接别名 data/bias。
 * 原始路径始终使用 data/bias + scale/mean/var 原始参数。
 */
typedef struct {
    int    size, channels, n, stride, groups, batchnorm, activate;
    float *data, *bias, *scale, *rolling_mean, *rolling_variance;
    float *data2, *bias2;
} FILTER;

/* 层类型枚举，顺序与 cfg 解析及 net_forward 分发一一对应 */
enum {
    LAYER_TYPE_CONV    ,
    LAYER_TYPE_AVGPOOL ,
    LAYER_TYPE_MAXPOOL ,
    LAYER_TYPE_UPSAMPLE,
    LAYER_TYPE_DROPOUT ,
    LAYER_TYPE_SHORTCUT,
    LAYER_TYPE_ROUTE   ,
    LAYER_TYPE_YOLO    ,
};

/*
 * 单层槽位。layer_list[i] 描述第 i 层自身参数；其输出尺寸写在
 * layer_list[i+1].matrix 中（由 cfg 解析阶段顺推填好），即第 i 层的"输出矩阵"
 * 实际是下一槽位的 matrix。depend_list 记录 route/shortcut 引用的源层下标。
 * yolo 层额外使用 class_num / anchor_list 等字段。
 */
typedef struct {
    int     type, refcnt;          /* refcnt: 被 route/shortcut 引用的计数，仅供 dump 展示 */
    MATRIX  matrix;
    FILTER  filter;
    int     depend_list[4];        /* route 最多 4 路、shortcut 1 路依赖 */
    int     depend_num;

    int     class_num;             /* ---- 以下为 yolo 层专用 ---- */
    int     anchor_num;
    int     anchor_list[9][2];     /* 经 mask 筛选后的该层锚框 w/h */
    float   ignore_thres, scale_x_y;
} LAYER;

/* 检测框。type 为类别下标，坐标在 NMS 阶段已从网络输入尺度还原到原图尺度 */
typedef struct {
    int   type;
    float score, x1, y1, x2, y2;
} BBOX;

/*
 * 整网描述与全局缓冲。
 * - bbox_list 直接别名 layer_list[0].matrix.data（复用输入图像缓冲存检测框，
 *   推理后段输入缓冲已无其他读者）；bbox_max 为该缓冲能容纳的最大框数。
 * - conv_impl : 卷积内核实现选择，CONV_IMPL_ORIG / _OPT / _AVX2（见 kernels/kernel_common.h）。
 *               avx2 在编译期缺失或 CPU 不支持时由派发层自动降级到 opt。
 *               opt / avx2 只认 BN 折叠后的 data2/bias2。
 * - bn_fused  : BN 是否已折叠进权重副本；切换实现时由 net_set_conv_impl 清零复判。
 * - weight_buf/foldbuf/layerbuf/cnntempbuf 均在 net_load 一次分配，net_free 统一释放。
 */
typedef struct {
    LAYER *layer_list;
    int    layer_num;
    BBOX  *bbox_list;
    int    bbox_num;
    int    bbox_max;
    int    s1, s2;
    int    conv_impl;
    int    bn_fused;
    int    weight_size;              /* 权重总元素数（float 个数，非字节） */
    float *weight_buf;               /* 原始权重（darknet 文件原样内容）   */
    float *foldbuf;                  /* BN 折叠后的 W'/b' 副本            */
    float *layerbuf;                 /* 所有层输出缓冲所在的 arena         */
    float *cnntempbuf;               /* 卷积临时缓冲（im2col 行块）        */
    int    cnnbufsize;               /* cnntempbuf 容量（float 数）        */
} NET;

/* 公共 API：加载 / 释放 / 写入输入图像 / 前向推理 / 打印网络结构 */
NET* net_load   (char *file1, char *file2);
void net_free   (NET *net);
void net_input  (NET *net, unsigned char *bgr, int w, int h, float *mean, float *norm);
void net_forward(NET *net);

/* 卷积实现切换：0=orig（原始标量） 1=opt（优化标量） 2=avx2（向量化） */
int  net_set_conv_impl     (NET *net, int impl);
int  net_effective_conv_impl(NET *net);
void net_dump   (NET *net);

#endif
