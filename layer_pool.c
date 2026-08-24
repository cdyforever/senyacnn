/*
 * layer_pool.c - 池化与上采样
 *
 * avgpool / maxpool 共用一个实现，flag 区分：1=取 max，0=取平均；
 * upsample 为最近邻放大。
 * 池化窗口允许越过输入右/下边界：max 时边界外的值不参与比较，
 * avg 时除数固定为 size*size（历史行为，与 darknet 一致）。
 */
#include "internal.h"

/*
 * 单个池化窗口计算。x,y 为窗口左上角在输入缓冲（含 pad）中的坐标，
 * mw/mh 为含 pad 的行宽/行高。flag: 1=max, 0=avg。
 */
static float filter_avgmax(float *mat, int mw, int mh, int x, int y, int fsize, int flag)
{
    float val; int i, j;
    mat += y * mw + x;
    val  = flag ? *mat : 0;                 /* max 用首元素做初值；avg 从 0 累加 */
    for (i=0; i<fsize; i++) {
        for (j=0; j<fsize; j++,mat++) {
            if (j + x < mw && i + y < mh) { /* 越过缓冲边界的元素跳过 */
                if (flag) {
                    if (val < *mat) val = *mat;
                } else {
                    val += *mat;
                }
            }
        }
        mat += mw - j;                      /* 行尾回退差值对齐下一行 */
    }
    return flag ? val : val / (fsize * fsize);
}

/* 通用池化：按 stride 在内部区域滑动窗口，逐通道处理 */
void layer_avgmaxpool_forward(LAYER *ilayer, LAYER *olayer, int flag)
{
    int  i, ix, iy, ox, oy, mwi, mhi, mwo;
    float *datai, *datao;
    mwi   = ilayer->matrix.width + ilayer->matrix.pad * 2;
    mhi   = ilayer->matrix.height+ ilayer->matrix.pad * 2;
    mwo   = olayer->matrix.width + olayer->matrix.pad * 2;
    datai = ilayer->matrix.data + ilayer->matrix.pad * mwi + ilayer->matrix.pad;   /* 输入内部左上角 */
    datao = olayer->matrix.data + olayer->matrix.pad * mwo + olayer->matrix.pad;   /* 输出内部左上角 */
    for (i=0; i<ilayer->matrix.channels; i++) {
        for (iy=0,oy=0; iy<ilayer->matrix.height; iy+=ilayer->filter.stride,oy++) {
            for (ix=0,ox=0; ix<ilayer->matrix.width; ix+=ilayer->filter.stride,ox++) {
                *datao++ = filter_avgmax(datai, mwi, mhi, ix, iy, ilayer->filter.size, flag);
            }
            datao += mwo - ox;                  /* 行尾对齐到输出缓冲下一行 */
        }
        datai += (ilayer->matrix.height + ilayer->matrix.pad * 2) * mwi;   /* 下一通道平面 */
        datao += olayer->matrix.pad * 2 * mwo;
    }
}

/* 最近邻上采样：out(x,y) = in(x/stride, y/stride)，通道数不变 */
void layer_upsample_forward(LAYER *ilayer, LAYER *olayer)
{
    int    mwi   = ilayer->matrix.width + ilayer->matrix.pad * 2;
    int    mwo   = olayer->matrix.width + olayer->matrix.pad * 2;
    float *datai = ilayer->matrix.data + ilayer->matrix.pad * mwi + ilayer->matrix.pad;
    float *datao = olayer->matrix.data + olayer->matrix.pad * mwo + olayer->matrix.pad;
    int    i, x, y;
    for (i=0; i<ilayer->matrix.channels; i++) {
        for (y=0; y<olayer->matrix.height; y++) {
            for (x=0; x<olayer->matrix.width; x++) {
                *datao++ = datai[(y / ilayer->filter.stride) * mwi + (x / ilayer->filter.stride)];
            }
            datao += mwo - x;
        }
        datai += (ilayer->matrix.height + ilayer->matrix.pad * 2) * mwi;
        datao += olayer->matrix.pad * 2 * mwo;
    }
}
