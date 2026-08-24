/*
 * layer_conv.c - 卷积层前向实现（原始路径 + 优化路径）
 *
 * 原始路径 layer_groupconv_forward：每个输出像素做一次 im2row 拷贝，随后对每个
 *   输出通道做点积，BN/激活逐元素应用。保留它作为逐位可复现的基准实现。
 * 优化路径 layer_groupconv_forward_opt：按层几何分派到三个内核——
 *   depthwise 直接窗口卷积 / 1x1 行累加 / 通用 im2col+GEMM；
 *   统一使用 BN 折叠后的 data2/bias2，内循环不含任何 BN 运算。
 *
 * 关键坐标约定（曾在此踩坑）：窗口锚点直接用 padded 数组坐标，
 * 即 im2row(matrix, x, y, ...) 的窗口覆盖 padded[y..y+size-1][x..x+size-1]。
 * 由于内部像素 (ix,iy) 对应 padded 坐标恰好是 (ix,iy)（pad 已含在数组里），
 * 这等价于"以内部像素为中心的 size 窗口"，与 darknet 语义一致。
 * 任何新内核都必须遵守同一约定，否则整体偏移一个像素且边界越界读。
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "internal.h"

/* 激活函数。linear 即恒等；sigmoid 仅由 yolo 解码使用 */
float activate(float x, int type)
{
    switch (type) {
    case ACTIVATE_TYPE_RELU   : return x > 0 ? x : 0;
    case ACTIVATE_TYPE_LEAKY  : return x > 0 ? x : 0.1f * x;
    case ACTIVATE_TYPE_SIGMOID: return 1.0f / (1.0f + (float)exp(-x));
    default: return x;
    }
}

/*
 * 快速平方根倒数（Quake III 魔数算法），一次牛顿迭代后误差约 0.2%。
 * 原始卷积路径在每个输出元素上调用它归一化 BN 方差；优化路径改用
 * 加载时折叠（精确 sqrtf），因此该函数仅剩原始路径在用。
 */
static float fast_inverse_sqrt(float x)
{
    union { float f; int32_t i; } fui;
    float halfx = 0.5f * x;
    fui.f = x;
    fui.i = 0x5F3759DF - (fui.i >> 1);
    x     = fui.f;
    x     = x * (1.5f - halfx * x * x);
    return x;
}

/*
 * im2row：把以 padded 坐标 (x,y) 为左上角的 size x size x channels 邻域
 * 展平拷贝到 buf，排列为 channel-major、行内 fy->fx，与权重布局
 * [n][channels][fy][fx] 的内两维一致，使点积按下标顺序一一对应。
 * 注意锚点不做 +pad 修正（见文件头"关键坐标约定"）。
 */
static void im2row(MATRIX *matrix, int x, int y, int fsize, float *buf)
{
    float *data = matrix->data +  y * (matrix->width + matrix->pad * 2) + x;
    int    i, j, k;
    for (i=0; i<matrix->channels; i++) {
        for (j=0; j<fsize; j++) {
            for (k=0; k<fsize; k++) *buf++ = *data++;
            /* 内层走完后 k==fsize，补回行尾差值，使每小行走满一行宽 */
            data += matrix->width + matrix->pad * 2 - k;
        }
        /* 回到本通道窗口首行起点，再跳到下一通道平面 */
        data -= (matrix->width + matrix->pad * 2) * fsize;
        data += (matrix->width + matrix->pad * 2) * (matrix->height + matrix->pad * 2);
    }
}

/*
 * 原始分组卷积实现（基准路径，保持历史行为逐位不变）。
 * 结构：每组 -> 每个输出位置 im2row 一次 -> 组内各输出通道共享这行数据做点积。
 * 性能短板（也是优化路径的出发点）：
 *   - BN 的 rsqrt 在每个输出元素上重复计算；
 *   - 输出写入按整平面大步长跳跃，cache 不友好。
 */
void layer_groupconv_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    MATRIX mati  = ilayer->matrix;
    MATRIX mato  = olayer->matrix;
    FILTER filter= ilayer->filter;
    int    ftotal, ix, iy, ox, oy, i, j, n;
    float *datao, *dataf;

    /* 把总量按组拆成"每组的输入/输出通道数与滤波器数" */
    mati.channels /= filter.groups;
    mato.channels /= filter.groups;
    filter.n      /= filter.groups;
    ftotal         = filter.size * filter.size * filter.channels;
    if (net->cnnbufsize < ftotal) {
        net->cnnbufsize = ftotal;
        if (net->cnntempbuf) free(net->cnntempbuf);
        net->cnntempbuf = malloc(net->cnnbufsize * sizeof(float));
        if (net->cnntempbuf == NULL) { printf("failed to allocate memory for cnntempbuf !"); return; }
    }

    for (n=0; n<filter.groups; n++) {
        for (iy=0,oy=0; iy<mati.height; iy+=filter.stride,oy++) {
            for (ix=0,ox=0; ix<mati.width; ix+=filter.stride,ox++) {
                im2row(&mati, ix, iy, filter.size, net->cnntempbuf);
                /* 该像素在第 i 个输出通道平面上的地址；i 循环内按整平面步进 */
                datao = mato.data + (mato.pad + oy) * (mato.width + mato.pad * 2) + mato.pad + ox;
                for (dataf=filter.data,i=0; i<filter.n; i++) {
                    /* 单链浮点累加：顺序固定，保证与历史基线逐位一致 */
                    for (*datao=0,j=0; j<ftotal; j++) *datao += *dataf++ * net->cnntempbuf[j];
                    if (filter.batchnorm) {
                        *datao = (*datao - filter.rolling_mean[i]) * fast_inverse_sqrt(filter.rolling_variance[i] + 0.00001f);
                        *datao*= filter.scale[i];
                    }
                   *datao  = activate(*datao + filter.bias[i], filter.activate);
                    datao += (mato.width + mato.pad * 2) * (mato.height + mato.pad * 2);
                }
            }
        }
        /* 各指针推进到下一组的切片 */
        mati.data  += (mati.width + mati.pad * 2) * (mati.height + mati.pad * 2) * mati.channels;
        mato.data  += (mato.width + mato.pad * 2) * (mato.height + mato.pad * 2) * mato.channels;
        filter.data+= filter.size * filter.size * filter.channels * filter.n;
        filter.bias            += filter.n;
        filter.scale           += filter.n;
        filter.rolling_mean    += filter.n;
        filter.rolling_variance+= filter.n;
    }
}

/* 优化内核用的激活分派：relu/leaky 内联为无分支表达式，其余转调通用 activate */
static __inline float actv(float x, int type)
{
    switch (type) {
    case ACTIVATE_TYPE_RELU : return x > 0 ? x : 0;
    case ACTIVATE_TYPE_LEAKY: return x > 0 ? x : 0.1f * x;
    default                 : return activate(x, type);
    }
}

/*
 * dot4：4 路独立累加器的点积。单链累加受 FP 加法延迟(~4 cycle)限制，
 * 4 条独立链让流水线打满，典型提速 1.5~2 倍。
 * 注意：求和顺序与单链不同 -> 与原始路径存在正常浮点差异（容差比对）。
 * 结果仍是确定性的：(a0+a1)+(a2+a3)，便于回归测试。
 */
static float dot4(const float * restrict w, const float * restrict r, int n)
{
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0; int j;
    for (j=0; j+4<=n; j+=4) {
        a0 += w[j  ] * r[j  ];
        a1 += w[j+1] * r[j+1];
        a2 += w[j+2] * r[j+2];
        a3 += w[j+3] * r[j+3];
    }
    for (; j<n; j++) a0 += w[j] * r[j];      /* 尾部并入 a0，保持确定性 */
    return (a0 + a1) + (a2 + a3);
}

/*
 * depthwise 卷积专用内核（groups == 输入通道数，每通道一组 3x3 类窗口）。
 * 相比通用路径省掉 im2row 拷贝——depthwise 时拷贝量与计算量相当。
 * 直接在含 pad 边缘的输入缓冲上滑窗：边缘已在加载时填好（0 或 -FLT_MAX），
 * 无需任何边界分支。窗口锚点用 padded 坐标（见文件头约定），与 im2row 一致。
 * 输出只写内部区域（含下一层的 pad 边缘偏移），边缘区保持加载时的填充值。
 */
static void conv_depthwise(MATRIX * restrict mati, MATRIX * restrict mato, const float * restrict W, const float * restrict B, int size, int stride, int act)
{
    int mwi = mati->width  + mati->pad * 2;   /* 输入缓冲行宽（含左右 pad） */
    int mhi = mati->height + mati->pad * 2;
    int mwo = mato->width  + mato->pad * 2;   /* 输出缓冲行宽               */
    int mho = mato->height + mato->pad * 2;
    int c, ky, kx, ox, oy;
    for (c=0; c<mati->channels; c++) {
        const float *w   = W    + c * size * size;                                  /* 本通道的核       */
        float       *out = mato->data + c * mwo * mho + mato->pad * mwo + mato->pad; /* 本通道内部左上角 */
        const float *in  = mati->data + c * mwi * mhi;                              /* 本通道平面基址   */
        for (oy=0; oy<mato->height; oy++) {
            for (ox=0; ox<mato->width; ox++) {
                const float *ip = in + (oy * stride) * mwi + ox * stride;
                float acc = 0;
                for (ky=0; ky<size; ky++, ip += mwi) {
                    for (kx=0; kx<size; kx++) acc += w[ky * size + kx] * ip[kx];
                }
                out[oy * mwo + ox] = actv(acc + B[c], act);
            }
        }
    }
}

/*
 * 1x1 步长1卷积专用内核（pointwise，MobileNetV2 类网络的主要算力所在）。
 * 无需 im2row：把每个输出行先初始化为 bias，再按输入通道逐个做
 * "标量乘 + 行加"，行常驻 L1，输入按行连续读取。
 * 输出槽位的 pad 属于下一层（可能非 0），故写入时统一带 margin 处理。
 * 激活必须在全部通道累加完成后一次性施加。
 */
static void conv_1x1(MATRIX * restrict mati, MATRIX * restrict mato, const float * restrict W, const float * restrict B, int act)
{
    int C     = mati->channels;             /* 输入通道数（groups==1）          */
    int wi    = mati->width;                /* 输入行宽：本内核要求 pad==0      */
    int plane = wi * mati->height;          /* 单通道平面大小                   */
    int mwo   = mato->width + mato->pad * 2;
    int i, c, x, y;
    const float *in = mati->data;
    for (i=0; i<mato->channels; i++) {
        const float *w   = W + i * C;       /* 第 i 个输出通道的 C 个系数       */
        float       *out = mato->data + i * (mwo * (mato->height + mato->pad * 2)) + mato->pad * mwo + mato->pad;
        for (y=0; y<mato->height; y++) {
            float *orow = out + y * mwo;
            for (x=0; x<mato->width; x++) orow[x] = B[i];                 /* 行初始化为 bias */
            for (c=0; c<C; c++) {
                float       wc  = w[c];
                const float *src = in + c * plane + y * wi;               /* 输入第 c 通道同行 */
                for (x=0; x<mato->width; x++) orow[x] += wc * src[x];     /* 累加顺序 c 升序   */
            }
        }
        switch (act) {
        case ACTIVATE_TYPE_RELU:
            for (y=0; y<mato->height; y++) {
                float *orow = out + y * mwo;
                for (x=0; x<mato->width; x++) { float t = orow[x]; orow[x] = t > 0 ? t : 0; }
            }
            break;
        case ACTIVATE_TYPE_LEAKY:
            for (y=0; y<mato->height; y++) {
                float *orow = out + y * mwo;
                for (x=0; x<mato->width; x++) { float t = orow[x]; orow[x] = t > 0 ? t : 0.1f * t; }
            }
            break;
        default:
            for (y=0; y<mato->height; y++) {
                float *orow = out + y * mwo;
                for (x=0; x<mato->width; x++) orow[x] = activate(orow[x], act);
            }
            break;
        }
    }
}

/*
 * 优化版分组卷积入口：按层几何选择内核。
 *   depthwise : groups == 输入通道数 且 每组输入通道数为 1
 *   1x1       : size==1 && stride==1 && 输入 pad==0 && groups==1
 *   其余       : 整组 im2col + 滤波器主序 GEMM 式遍历
 * 三条路径都要求 BN 已折叠（bn_fused），因为它们只认 data2/bias2。
 */
void layer_groupconv_forward_opt(NET *net, LAYER *ilayer, LAYER *olayer)
{
    MATRIX mati  = ilayer->matrix;
    MATRIX mato  = olayer->matrix;
    FILTER filter= ilayer->filter;
    int    is_dw, is_1x1;
    int    ftotal, scratch, mwo, ix, iy, ox, oy, i, x, y, n;
    float *col, *datao, *dataf;

    is_dw  = (filter.channels == 1 && filter.groups == ilayer->matrix.channels && filter.groups >= 1);
    is_1x1 = (filter.size == 1 && filter.stride == 1 && ilayer->matrix.pad == 0 && filter.groups == 1);

    if (is_dw) {
        if (!net->bn_fused && filter.batchnorm) { printf("bn not fused!\n"); return; }
        conv_depthwise(&mati, &mato, filter.data2, filter.bias2, filter.size, filter.stride, filter.activate);
        return;
    }
    if (is_1x1) {
        if (!net->bn_fused && filter.batchnorm) { printf("bn not fused!\n"); return; }
        conv_1x1(&mati, &mato, filter.data2, filter.bias2, filter.activate);
        return;
    }

    /* ---- 通用路径：im2col 一次构建整组列矩阵，再按滤波器主序遍历 ---- */
    mati.channels /= filter.groups;
    mato.channels /= filter.groups;
    filter.n      /= filter.groups;
    ftotal         = filter.size * filter.size * filter.channels;
    scratch        = ftotal * mato.width * mato.height;
    if (net->cnnbufsize < scratch) {
        net->cnnbufsize = scratch;
        if (net->cnntempbuf) free(net->cnntempbuf);
        net->cnntempbuf = malloc(net->cnnbufsize * sizeof(float));
        if (net->cnntempbuf == NULL) { printf("failed to allocate memory for cnntempbuf !"); return; }
    }

    col = net->cnntempbuf;
    mwo = mato.width + mato.pad * 2;

    for (n=0; n<filter.groups; n++) {
        /* 构建本组全部输出像素对应的 im2col 行（拷贝总量与原始路径相同：
         * 每输出像素一次），之后所有滤波器复用这份只读数据 */
        for (iy=0,oy=0; iy<mati.height; iy+=filter.stride,oy++) {
            for (ix=0,ox=0; ix<mati.width; ix+=filter.stride,ox++) {
                im2row(&mati, ix, iy, filter.size, col + ((oy * mato.width) + ox) * ftotal);
            }
        }
        /* 滤波器在外 -> 输出按行连续写（对比原始路径的大步长跳跃写）；
         * 点积经 dot4 展开，BN 已折叠进 data2/bias2 */
        for (dataf=filter.data2,i=0; i<filter.n; i++,dataf+=ftotal) {
            float bias  = filter.bias2[i];
            datao = mato.data + (mato.pad * mwo + mato.pad) + i * (mwo * (mato.height + mato.pad * 2));
            switch (filter.activate) {
            case ACTIVATE_TYPE_RELU:
                for (y=0; y<mato.height; y++, datao += mato.pad * 2) {
                    for (x=0; x<mato.width; x++) {
                        float t = dot4(dataf, col + ((y * mato.width) + x) * ftotal, ftotal) + bias;
                        *datao++ = t > 0 ? t : 0;
                    }
                }
                break;
            case ACTIVATE_TYPE_LEAKY:
                for (y=0; y<mato.height; y++, datao += mato.pad * 2) {
                    for (x=0; x<mato.width; x++) {
                        float t = dot4(dataf, col + ((y * mato.width) + x) * ftotal, ftotal) + bias;
                        *datao++ = t > 0 ? t : 0.1f * t;
                    }
                }
                break;
            default:
                for (y=0; y<mato.height; y++, datao += mato.pad * 2) {
                    for (x=0; x<mato.width; x++) {
                       *datao++ = activate(dot4(dataf, col + ((y * mato.width) + x) * ftotal, ftotal) + bias, filter.activate);
                    }
                }
                break;
            }
        }
        /* 推进到下一组切片（data/data2 同步推进，bias/bias2 同理） */
        mati.data  += (mati.width + mati.pad * 2) * (mati.height + mati.pad * 2) * mati.channels;
        mato.data  += (mato.width + mato.pad * 2) * (mato.height + mato.pad * 2) * mato.channels;
        filter.data += ftotal * filter.n;
        filter.data2+= ftotal * filter.n;
        filter.bias            += filter.n;
        filter.bias2           += filter.n;
        filter.scale           += filter.n;
        filter.rolling_mean    += filter.n;
        filter.rolling_variance+= filter.n;
    }
}
