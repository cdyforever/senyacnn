/*
 * layer_yolo.c - YOLO 检测头解码
 *
 * 把 [yolo] 层的原始输出张量解码成候选框并写入 net->bbox_list。
 * 张量通道布局（每个锚框一组，共 anchor 组）：
 *   [tx ty tw th to pc0 pc1 ... pc(class_num-1)]
 * 解码公式：
 *   bx = (col + sigmoid(tx)) / grid_w          （再乘网络输入宽得像素坐标）
 *   by = (row + sigmoid(ty)) / grid_h
 *   bw = exp(tw) * anchor_w * scale_x_y
 *   bh = exp(th) * anchor_h * scale_x_y
 *   score = sigmoid(objectness) >= ignore_thresh 才生成候选
 *
 * 已知局限（历史行为，保留待修）：每组锚框数硬编码为 3（k<3 循环），
 * 未使用解析得到的 anchor_num。
 */
#include <math.h>
#include "internal.h"

/* CHW 布局取值：(w,h,c) -> data[c*平面大小 + h*行宽 + w] */
static float get_matrix_data(MATRIX *mat, int w, int h, int c)
{
    int mw = mat->width  + mat->pad * 2;
    int mh = mat->height + mat->pad * 2;
    return mat->data[c * mw * mh + h * mw + w];
}

/* 逐网格、逐锚框扫描，先看 objectness 再找类别最大值（提前剪枝省 exp/sigmoid） */
void layer_yolo_forward(NET *net, LAYER *ilayer)
{
    int i, j, k, l; float confidence;
    for (i=0; i<ilayer->matrix.height; i++) {           /* 网格行 */
        for (j=0; j<ilayer->matrix.width; j++) {        /* 网格列 */
            for (k=0; k<3; k++) {                       /* 锚框序号（历史行为：固定 3 个） */
                int dstart = k * (4 + 1 + ilayer->class_num), cindex = 0;
                float bs = get_matrix_data(&ilayer->matrix, j, i, dstart + 4);   /* objectness */
                float cs = get_matrix_data(&ilayer->matrix, j, i, dstart + 5);   /* 类别0 分数 */
                /* 找最大类别分数 */
                for (l=1; l<ilayer->class_num; l++) {
                    float val = get_matrix_data(&ilayer->matrix, j, i, dstart + 5 + l);
                    if (cs < val) { cs = val; cindex = l; }
                }
                confidence = activate(bs, ACTIVATE_TYPE_SIGMOID);
                if (confidence >= ilayer->ignore_thres) {
                    float tx = get_matrix_data(&ilayer->matrix, j, i, dstart + 0);
                    float ty = get_matrix_data(&ilayer->matrix, j, i, dstart + 1);
                    float tw = get_matrix_data(&ilayer->matrix, j, i, dstart + 2);
                    float th = get_matrix_data(&ilayer->matrix, j, i, dstart + 3);
                    /* 中心点：网格坐标 + sigmoid 偏移，缩放到网络输入分辨率 */
                    float bbox_cx = (j + activate(tx, ACTIVATE_TYPE_SIGMOID)) * net->layer_list[0].matrix.width / ilayer->matrix.width ;
                    float bbox_cy = (i + activate(ty, ACTIVATE_TYPE_SIGMOID)) * net->layer_list[0].matrix.height/ ilayer->matrix.height;
                    /* 宽高：exp(t) 相对锚框的缩放 */
                    float bbox_w  = (float)exp(tw) * ilayer->anchor_list[k][0] * ilayer->scale_x_y;
                    float bbox_h  = (float)exp(th) * ilayer->anchor_list[k][1] * ilayer->scale_x_y;
                    /* 写入 bbox_list（复用输入图像缓冲，容量见 bbox_max） */
                    if (net->bbox_num < net->bbox_max) {
                        net->bbox_list[net->bbox_num].type  = cindex;
                        net->bbox_list[net->bbox_num].score = activate(cs, ACTIVATE_TYPE_SIGMOID);;
                        net->bbox_list[net->bbox_num].x1    = bbox_cx - bbox_w * 0.5f;
                        net->bbox_list[net->bbox_num].y1    = bbox_cy - bbox_h * 0.5f;
                        net->bbox_list[net->bbox_num].x2    = bbox_cx + bbox_w * 0.5f;
                        net->bbox_list[net->bbox_num].y2    = bbox_cy + bbox_h * 0.5f;
                        net->bbox_num++;
                    }
                }
            }
        }
    }
}
