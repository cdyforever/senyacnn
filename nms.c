/*
 * nms.c - 非极大值抑制后处理
 *
 * 流程：按分数降序排序 -> 同类别两两算 IoU，超过阈值者置 0 -> 压缩数组
 * 并把坐标从网络输入尺度还原到原图尺度。
 */
#include <stdlib.h>
#include <string.h>
#include "internal.h"

/* qsort 比较器：分数大的排前面 */
static int bbox_cmp(const void *p1, const void *p2)
{
    if      (((BBOX*)p1)->score < ((BBOX*)p2)->score) return  1;
    else if (((BBOX*)p1)->score > ((BBOX*)p2)->score) return -1;
    else return 0;
}

/*
 * bboxlist/n : 候选框数组与数量（score==0 的槽位视为已删除）
 * threshold  : IoU 判重阈值
 * min        : 1 = IoU 分母取较小框面积（min 模式，常用于人脸这类小目标重叠），
 *              0 = 标准交并比
 * s1/s2      : letterbox 缩放比例 s1/s2，输出坐标乘以该比值还原到原图
 * 返回值     : 抑制后剩余的框数
 */
int nms(BBOX *bboxlist, int n, float threshold, int min, int s1, int s2)
{
    int i, j, c;
    if (!bboxlist || !n) return 0;
    qsort(bboxlist, n, sizeof(BBOX), bbox_cmp);
    /* 外层 i 为当前"存活基准框"；内层扫其后所有存活框，同类别且重叠大者淘汰。
     * 用 i==-1 作哨兵：每轮结束指向下一个未处理的存活框。 */
    for (i=0; i<n && i!=-1; ) {
        for (c=i,j=i+1,i=-1; j<n; j++) {
            if (bboxlist[j].score == 0) continue;      /* 已被抑制 */
            if (bboxlist[c].type == bboxlist[j].type) {
                float xc1, yc1, xc2, yc2, sc, s1, s2, ss, iou;
                /* 交集矩形 */
                xc1 = bboxlist[c].x1 > bboxlist[j].x1 ? bboxlist[c].x1 : bboxlist[j].x1;
                yc1 = bboxlist[c].y1 > bboxlist[j].y1 ? bboxlist[c].y1 : bboxlist[j].y1;
                xc2 = bboxlist[c].x2 < bboxlist[j].x2 ? bboxlist[c].x2 : bboxlist[j].x2;
                yc2 = bboxlist[c].y2 < bboxlist[j].y2 ? bboxlist[c].y2 : bboxlist[j].y2;
                sc  = (xc1 < xc2 && yc1 < yc2) ? (xc2 - xc1) * (yc2 - yc1) : 0;
                s1  = (bboxlist[c].x2 - bboxlist[c].x1) * (bboxlist[c].y2 - bboxlist[c].y1);
                s2  = (bboxlist[j].x2 - bboxlist[j].x1) * (bboxlist[j].y2 - bboxlist[j].y1);
                ss  = s1 + s2 - sc;                    /* 并集面积 */
                if (min) iou = sc / (s1 < s2 ? s1 : s2);
                else     iou = sc / ss;
                if (iou > threshold) bboxlist[j].score = 0;   /* 抑制低分重叠框 */
                else if (i == -1) i = j;
            } else if (i == -1) i = j;
        }
    }
    /* 压缩存活框到数组头部，同时把坐标乘 s1/s2 还原到原图尺度 */
    for (i=0,j=0; i<n; i++) {
        if (bboxlist[i].score) {
            bboxlist[j  ].score= bboxlist[i].score;
            bboxlist[j  ].type = bboxlist[i].type;
            bboxlist[j  ].x1   = bboxlist[i].x1 * s1 / s2;
            bboxlist[j  ].y1   = bboxlist[i].y1 * s1 / s2;
            bboxlist[j  ].x2   = bboxlist[i].x2 * s1 / s2;
            bboxlist[j++].y2   = bboxlist[i].y2 * s1 / s2;
        }
    }
    memset(bboxlist + j, 0, sizeof(BBOX) * (n - j));
    return j;
}
