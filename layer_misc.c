/*
 * layer_misc.c - dropout / shortcut / route 三种"无参数"层
 *
 * dropout: 推理期恒等映射，直接让输出槽位别名输入缓冲（arena 内共享安全）；
 * shortcut: 逐元素相加 depend_list[0] 指向的源层，再过激活函数；
 * route:   把若干源层在通道维拼接（逐行 memcpy）。
 */
#include <string.h>
#include "internal.h"

/*
 * 推理期 dropout 等价于恒等映射。这里让输出槽位与输入槽位指向同一块
 * arena 内存（只读共享）：后续层把 olayer 当普通输入读取即可。
 * 注意不要在此释放或置空任何指针——所有权统一归 arena。
 */
void layer_dropout_forward(LAYER *ilayer, LAYER *olayer)
{
    olayer->matrix.data = ilayer->matrix.data;
}

/* 残差加法：out = act(in1 + in2)，in2 来自 from=-k 解析出的 depend_list[0] */
void layer_shortcut_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    LAYER  *slayer = net->layer_list + ilayer->depend_list[0] + 1;   /* 被加的源层 */
    int     mwo= olayer->matrix.width + olayer->matrix.pad * 2;
    int     mw1= ilayer->matrix.width + ilayer->matrix.pad * 2;
    int     mw2= slayer->matrix.width + slayer->matrix.pad * 2;
    float  *datao = olayer->matrix.data + olayer->matrix.pad * mwo + olayer->matrix.pad;
    float  *data1 = ilayer->matrix.data + ilayer->matrix.pad * mw1 + ilayer->matrix.pad;
    float  *data2 = slayer->matrix.data + slayer->matrix.pad * mw2 + slayer->matrix.pad;
    int     i, x, y;
    for (i=0; i<olayer->matrix.channels; i++) {
        for (y=0; y<olayer->matrix.height; y++) {
            for (x=0; x<olayer->matrix.width; x++) {
                *datao++ = activate(*data1++ + *data2++, ilayer->filter.activate);
            }
            /* 各自行尾回退差值（x==width），对齐到下一行行首 */
            datao += mwo - x;
            data1 += mw1 - x;
            data2 += mw2 - x;
        }
        /* 跳过输出左右 pad 与两路输入各自的左右 pad，进入下一通道 */
        datao += olayer->matrix.pad * 2 * mwo;
        data1 += ilayer->matrix.pad * 2 * mw1;
        data2 += slayer->matrix.pad * 2 * mw2;
    }
}

/* 通道拼接：依次把每一路源层的所有通道平面逐行 memcpy 到输出 */
void layer_route_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    int    mwo   = olayer->matrix.width + olayer->matrix.pad * 2;
    float *datao = olayer->matrix.data + olayer->matrix.pad * mwo + olayer->matrix.pad;
    int    i, j, k;
    for (i=0; i<ilayer->depend_num; i++) {
        LAYER *rlayer = net->layer_list + ilayer->depend_list[i] + 1;
        int    mwr    = rlayer->matrix.width + rlayer->matrix.pad * 2;
        float *datar  = rlayer->matrix.data + rlayer->matrix.pad * mwr + rlayer->matrix.pad;
        for (j=0; j<rlayer->matrix.channels; j++) {
            for (k=0; k<rlayer->matrix.height; k++) {
                memcpy(datao, datar, olayer->matrix.width * sizeof(float));
                datao += mwo; datar += mwr;
            }
            datao += olayer->matrix.pad * 2 * mwo;
            datar += rlayer->matrix.pad * 2 * mwr;
        }
    }
}
