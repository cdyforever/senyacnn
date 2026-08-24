/*
 * parser.c - cfg 网络解析 + weights 权重装载 + 缓冲预分配 + BN 折叠
 *
 * net_load 的整体流程：
 *   1. 把整个 .cfg 读入内存，逐段（[xxx] ... 到下一个 '[' 前）解析出各层参数；
 *      解析时顺推填充下一槽位的输出尺寸（layer_list[i+1].matrix）。
 *   2. 按 conv 层参数累计权重元素数，分配 weight_buf；按 BN 层规模分配 foldbuf。
 *   3. 按文件顺序给每个 conv 层的 bias/scale/mean/var/data 指针划片，
 *      再一次性读入权重文件（跳过 16 字节头）。BN 层同时在 foldbuf 中
 *      划出 data2/bias2 副本片区，非 BN 层让 data2/bias2 别名原始指针。
 *   4. 分配输入缓冲 slot0（同时复用作 bbox 存储区），以及全部层输出的 arena
 *      （layerbuf）和卷积临时缓冲（cnntempbuf，按最大需求一次到位）；
 *      各输出缓冲的 pad 边缘在此一次性填充，forward 阶段不再分配/填充。
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "internal.h"

/* 把整个文件读进以 '\0' 结尾的缓冲（cfg 是文本，可整体做字符串搜索） */
static char* load_file_to_buffer(char *file)
{
    FILE *fp = fopen(file, "rb");
    char *buf= NULL;
    int   size;
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buf  = malloc(size + 1);
    if (buf) { fread(buf, 1, size, fp); buf[size] = '\0'; }
    fclose(fp);
    return buf;
}

/*
 * 统计 cfg 中可识别的层段落数量（[net] 不算层）。
 * 用于一次 calloc 出 NET 结构 + layer_num+1 个 LAYER 槽位的连续内存，
 * 多出的 1 个槽位描述最后一层的输出。
 */
static int get_total_layers(char *str)
{
    static const char *LAYER_TYPE[] = { "[conv]", "[convolutional]", "[avg]", "[avgpool]", "[max]", "[maxpool]", "[upsample]", "[dropout]", "[shortcut]", "[route]", "[yolo]", NULL };
    int n = 0, i;
    while (str && (str = strstr(str, "["))) {
        for (i=0; LAYER_TYPE[i]; i++) {
            if (strstr(str, LAYER_TYPE[i]) == str) break;   /* 必须是段首前缀匹配，避免误认 */
        }
        if (LAYER_TYPE[i]) n++;
        str = strstr(str, "]");
    }
    return n;
}

/*
 * 在 [str, end) 区间内取 key=value 的 value 文本。
 * key 缺失时返回 NULL 且 val 置空串——调用方直接 atoi("") 得 0 作为缺省值。
 */
static char* parse_params(const char *str, const char *end, const char *key, char *val, int len)
{
    char *p = (char*)strstr(str, key); int i;

    *val = '\0';
    if (!p || (end && p >= end)) return NULL;
    p += strlen(key);
    if (*p == '\0') return NULL;

    /* 跳过 '=' 与空格，读到行尾即为 value */
    while (*p) {
        if (*p != '=' && *p != ' ') break;
        else p++;
    }

    for (i=0; i<len; i++) {
        if (*p == '\n' || *p == '\0') break;
        val[i] = *p++;
    }
    val[i < len ? i : len - 1] = '\0';
    return val;
}

/* 激活名 -> 枚举值，前缀匹配；未知返回 -1（forward 时按 linear 处理） */
static int get_activation_type_int(char *str)
{
    static const char *STR_TAB[] = { "linear", "relu", "leaky", NULL };
    int  i;
    for (i=0; STR_TAB[i]; i++) {
        if (strstr(str, STR_TAB[i]) == str) return i;
    }
    return -1;
}

char* get_activation_type_string(int type)
{
    static const char *STR_TAB[] = { "linear", "relu", "leaky", NULL };
    return (type >= 0 && type <= 2) ? (char*)STR_TAB[type] : "unknown";
}

char* get_layer_type_string(int type)
{
    static const char *STR_TAB[] = { "conv", "avgpool", "maxpool", "upsample", "dropout", "shortcut", "route", "yolo" };
    return (type >= 0 && type <= 7) ? (char*)STR_TAB[type] : "unknown";
}

/*
 * 由本层参数推导输出几何并写入下一槽位 out：
 *   - pad 标志展开为实际像素 pad = size/2（darknet 语义，仅奇数核对称）
 *   - conv: out = (in - size + 2*pad)/stride + 1；池化类: out = in/stride
 * 注意：out->matrix.pad 保持不变——它属于"下一层自己的输入 pad"，
 * 在解析到下一层段落时才由本函数设置。因此运行期每层缓冲自带的
 * 边缘宽度恰好等于该层作为输入时所需的 pad，读写两端自然一致。
 */
static void calculate_output_whcp(LAYER *in, LAYER *out)
{
    in ->matrix.pad      = in->matrix.pad ? in->filter.size / 2 : 0;
    out->matrix.channels = in->filter.n;
    out->matrix.width    = in->type == LAYER_TYPE_CONV ? (in->matrix.width - in->filter.size + in->matrix.pad * 2) / in->filter.stride + 1 : in->matrix.width / in->filter.stride;
    out->matrix.height   = in->type == LAYER_TYPE_CONV ? (in->matrix.height- in->filter.size + in->matrix.pad * 2) / in->filter.stride + 1 : in->matrix.height/ in->filter.stride;
}

NET* net_load(char *fcfg, char *fweights)
{
    char   *cfgstr = load_file_to_buffer(fcfg), *pstart, *pend, strval[256];
    NET    *net    = NULL;
    MATRIX *mat    = NULL;
    int     layers, layercur = 0, i;
    if (!cfgstr) return NULL;

    /* NET 结构与层数组一次 calloc：零初始化省去逐字段缺省，尾部多留一个输出槽位 */
    layers = get_total_layers(cfgstr);
    net    = calloc(1, sizeof(NET) + (layers + 1) * sizeof(LAYER));
    pstart = cfgstr;
    net->layer_list = (LAYER*)((char*)net + sizeof(NET));
    net->layer_num  = layers;

    /* 逐段解析：pstart 指向当前 '['，pend 指向下一个 '[' 前（段边界） */
    while (pstart && (pstart = strstr(pstart, "["))) {
        if ((pend = strstr(pstart + 1, "["))) pend = pend - 1;
        if (strstr(pstart, "[net]") == pstart) {
            /* [net] 段定义网络输入尺寸，存入 slot0 的 matrix */
            parse_params(pstart, pend, "width"   , strval, sizeof(strval)); net->layer_list[0].matrix.width   = atoi(strval);
            parse_params(pstart, pend, "height"  , strval, sizeof(strval)); net->layer_list[0].matrix.height  = atoi(strval);
            parse_params(pstart, pend, "channels", strval, sizeof(strval)); net->layer_list[0].matrix.channels= atoi(strval);
        } else if (strstr(pstart, "[conv]") == pstart || strstr(pstart, "[convolutional]") == pstart) {
            parse_params(pstart, pend, "filters" , strval, sizeof(strval)); net->layer_list[layercur].filter.n     = atoi(strval);
            parse_params(pstart, pend, "size"    , strval, sizeof(strval)); net->layer_list[layercur].filter.size  = atoi(strval);
            parse_params(pstart, pend, "stride"  , strval, sizeof(strval)); net->layer_list[layercur].filter.stride= atoi(strval);
            parse_params(pstart, pend, "groups"  , strval, sizeof(strval)); net->layer_list[layercur].filter.groups= atoi(strval);
            parse_params(pstart, pend, "pad"     , strval, sizeof(strval)); net->layer_list[layercur].matrix.pad   = atoi(strval);
            parse_params(pstart, pend, "batch_normalize", strval, sizeof(strval)); net->layer_list[layercur].filter.batchnorm = atoi(strval);
            parse_params(pstart, pend, "activation"     , strval, sizeof(strval)); net->layer_list[layercur].filter.activate  = get_activation_type_int(strval);
            if (net->layer_list[layercur].filter.stride== 0) net->layer_list[layercur].filter.stride= 1;
            if (net->layer_list[layercur].filter.groups== 0) net->layer_list[layercur].filter.groups= 1;
            /* groups=1 即普通卷积；groups==输入通道即 depthwise，均由通用公式覆盖 */
            net->layer_list[layercur].filter.channels = net->layer_list[layercur].matrix.channels / net->layer_list[layercur].filter.groups;
            /* 权重元素数 = 卷积核总量 + 每输出通道 1 个 bias（BN 层再加 scale/mean/var 各 n 个） */
            net->weight_size += net->layer_list[layercur].filter.size * net->layer_list[layercur].filter.size * net->layer_list[layercur].filter.channels * net->layer_list[layercur].filter.n;
            net->weight_size += net->layer_list[layercur].filter.n * (1 + !!net->layer_list[layercur].filter.batchnorm * 3);
            net->layer_list[layercur++].type = LAYER_TYPE_CONV;
            calculate_output_whcp(net->layer_list + layercur - 1, net->layer_list + layercur);
        } else if (strstr(pstart, "[avg]") == pstart || strstr(pstart, "[avgpool]") == pstart || strstr(pstart, "[max]") == pstart || strstr(pstart, "[maxpool]") == pstart) {
            parse_params(pstart, pend, "size"  , strval, sizeof(strval)); net->layer_list[layercur].filter.size  = atoi(strval);
            parse_params(pstart, pend, "stride", strval, sizeof(strval)); net->layer_list[layercur].filter.stride= atoi(strval);
            /* 池化不改变通道数，借用 filter.n 存通道数供 forward 使用 */
            net->layer_list[layercur  ].filter.n = net->layer_list[layercur].matrix.channels;
            net->layer_list[layercur++].type = (strstr(pstart, "[avg") == pstart) ? LAYER_TYPE_AVGPOOL : LAYER_TYPE_MAXPOOL;
            calculate_output_whcp(net->layer_list + layercur - 1, net->layer_list + layercur);
        } else if (strstr(pstart, "[upsample]") == pstart) {
            parse_params(pstart, pend, "stride" , strval, sizeof(strval)); net->layer_list[layercur].filter.stride = atoi(strval);
            net->layer_list[layercur+1].matrix.channels = net->layer_list[layercur].matrix.channels;
            net->layer_list[layercur+1].matrix.width    = net->layer_list[layercur].matrix.width  * net->layer_list[layercur].filter.stride;
            net->layer_list[layercur+1].matrix.height   = net->layer_list[layercur].matrix.height * net->layer_list[layercur].filter.stride;
            net->layer_list[layercur++].type = LAYER_TYPE_UPSAMPLE;
        } else if (strstr(pstart, "[dropout]") == pstart || strstr(pstart, "[shortcut]") == pstart) {
            /* 推理期 dropout 是恒等映射，输出尺寸与输入相同 */
            net->layer_list[layercur + 1].matrix.channels = net->layer_list[layercur].matrix.channels;
            net->layer_list[layercur + 1].matrix.width    = net->layer_list[layercur].matrix.width;
            net->layer_list[layercur + 1].matrix.height   = net->layer_list[layercur].matrix.height;
            if (strstr(pstart, "[dropout]") == pstart) {
                net->layer_list[layercur++].type = LAYER_TYPE_DROPOUT;
            } else {
                /* shortcut: from=-k 相对下标转绝对下标；激活作用于加和结果 */
                parse_params(pstart, pend, "from"      , strval, sizeof(strval)); net->layer_list[layercur].depend_list[0] = atoi(strval) + layercur;
                parse_params(pstart, pend, "activation", strval, sizeof(strval)); net->layer_list[layercur].filter.activate= get_activation_type_int(strval);
                net->layer_list[layercur  ].depend_num = 1;
                net->layer_list[layercur++].type = LAYER_TYPE_SHORTCUT;
            }
        } else if (strstr(pstart, "[route]") == pstart) {
            char *str; int dep = 0;
            /* route 把若干层在通道维拼接：layers=-4,-1... 负数为相对下标 */
            parse_params(pstart, pend, "layers", strval, sizeof(strval));
            for (i=0; i<4 && (str = strtok(i ? NULL : strval, ",")); i++) {
                dep = atoi(str);
                dep = dep > 0 ? dep : layercur + dep;
                net->layer_list[layercur + 0].depend_list[i]   = dep;
                /* 输出通道数累加，宽高取最后一路（要求各路空间尺寸一致） */
                net->layer_list[layercur + 1].matrix.channels += net->layer_list[dep + 1].matrix.channels;
                net->layer_list[layercur + 1].matrix.width     = net->layer_list[dep + 1].matrix.width;
                net->layer_list[layercur + 1].matrix.height    = net->layer_list[dep + 1].matrix.height;
            }
            net->layer_list[layercur  ].depend_num = i;
            net->layer_list[layercur++].type = LAYER_TYPE_ROUTE;
        } else if (strstr(pstart, "[yolo]") == pstart) {
            char *str; int masks[9]; int anchors[9][2];
            parse_params(pstart, pend, "classes"      , strval, sizeof(strval)); net->layer_list[layercur].class_num   = atoi(strval);
            parse_params(pstart, pend, "scale_x_y"    , strval, sizeof(strval)); net->layer_list[layercur].scale_x_y   = (strcmp(strval, "") == 0) ? 1.0f : (float)atof(strval);
            parse_params(pstart, pend, "ignore_thresh", strval, sizeof(strval)); net->layer_list[layercur].ignore_thres= (float)atof(strval);
            /* mask 选出本层使用的锚框编号（指向 anchors 列表的下标） */
            parse_params(pstart, pend, "mask", strval, sizeof(strval));
            for (i=0; i<9 && (str = strtok(i ? NULL : strval, ",")); i++) masks[i] = atoi(str);
            net->layer_list[layercur].anchor_num = i;
            /* anchors 为成对的 w,h 序列 */
            parse_params(pstart, pend, "anchors", strval, sizeof(strval));
            for (i=0; i<9 && (str = strtok(i ? NULL : strval, ",")); i++) {
                anchors[i][0] = atoi(str);
                str = strtok(NULL, ",");
                anchors[i][1] = atoi(str);
            }
            /* 按 mask 提炼出本层的锚框表 */
            for (i=0; i<net->layer_list[layercur].anchor_num; i++) {
                net->layer_list[layercur].anchor_list[i][0] = anchors[masks[i]][0];
                net->layer_list[layercur].anchor_list[i][1] = anchors[masks[i]][1];
            }
            net->layer_list[layercur++].type = LAYER_TYPE_YOLO;
        }
        pstart = pend;
    }
    free(cfgstr);

    /* 统计 BN 折叠副本所需容量：每个 BN 卷积层需要 data2(n*ftot) + bias2(n) */
    {
        int fold = 0;
        for (i=0; i<layers; i++) {
            if (net->layer_list[i].type == LAYER_TYPE_CONV && net->layer_list[i].filter.batchnorm) {
                FILTER *f = &net->layer_list[i].filter;
                fold += f->size * f->size * f->channels * f->n + f->n;
            }
        }
        net->foldbuf = fold ? malloc(fold * sizeof(float)) : NULL;
        if (fold && !net->foldbuf) { printf("failed to allocate buffers for net_load !\n"); net_free(net); return NULL; }
    }

    net->weight_buf = malloc(net->weight_size * sizeof(float));
    if (net->weight_buf) {
        float *pfloat = net->weight_buf, *pfold = net->foldbuf; FILE *fp;
        /* 按运行期访问顺序给各 conv 层划片。darknet 权重文件顺序为
         * [bias][scale][mean][var][data]，与此处推进顺序一致，可直接整块读入 */
        for (i=0; i<layers; i++) {
            if (net->layer_list[i].type == LAYER_TYPE_CONV) {
                FILTER *filter = &net->layer_list[i].filter;
                int     ftot   = filter->size * filter->size * filter->channels * filter->n;
                filter->bias = pfloat; pfloat += filter->n;
                if (filter->batchnorm) {
                    filter->scale            = pfloat; pfloat += filter->n;
                    filter->rolling_mean     = pfloat; pfloat += filter->n;
                    filter->rolling_variance = pfloat; pfloat += filter->n;
                    /* BN 层在 foldbuf 中预留 data2/bias2 副本片区（见 net_bn_fold） */
                    filter->bias2 = pfold; pfold += filter->n;
                    filter->data2 = pfold; pfold += ftot;
                }
                filter->data = pfloat; pfloat += ftot;
                /* 非 BN 层无需副本：优化内核经 data2/bias2 访问时直接别名原指针 */
                if (!filter->batchnorm) { filter->bias2 = filter->bias; filter->data2 = filter->data; }
            }
        }
        if ((fp = fopen(fweights, "rb"))) { fseek (fp, sizeof(WEIGHTS_FILE_HEADER), SEEK_SET); fread (net->weight_buf, 1, net->weight_size * sizeof(float), fp); fclose(fp); }
    }

    /* slot0：网络输入缓冲。calloc 保证未写的区域（letterbox 留白、pad 边缘）全 0。
     * 该缓冲同时被复用为检测框数组 bbox_list——推理后段已无人读输入图像，
     * yolo 层把 BBOX 直接写在这里（见 layer_yolo.c）。 */
    mat           = &(net->layer_list[0].matrix);
    mat->data     = calloc(1, (mat->width + mat->pad * 2) * (mat->height + mat->pad * 2) * mat->channels * sizeof(float));
    net->bbox_max = (mat->width + mat->pad * 2) * (mat->height + mat->pad * 2) * mat->channels * sizeof(float) / sizeof(BBOX);
    net->bbox_list= (BBOX*)net->layer_list[0].matrix.data;

    /* ---- 缓冲预分配：所有中间结果一次性到位，forward 零堆操作 ---- */
    {
        int total = 0, off = 0, tmpmax = 0, size;
        /* 1) 所有输出槽位（1..layers）的平面总字节数 -> arena(layerbuf) 容量 */
        for (i=1; i<=layers; i++) {
            MATRIX *m = &net->layer_list[i].matrix;
            size = (m->width + m->pad * 2) * (m->height + m->pad * 2) * m->channels;
            total += size;
        }
        /* 2) 卷积临时缓冲上限：原始路径需 ftot（单行 im2row），
         *    优化路径需 ftot*ow*oh（整组 im2col），取两者最大者一次分配 */
        for (i=0; i<layers; i++) {
            if (net->layer_list[i].type == LAYER_TYPE_CONV) {
                FILTER *f = &net->layer_list[i].filter;
                MATRIX *m = &net->layer_list[i+1].matrix;
                int ft   = f->size * f->size * f->channels;
                int need = ft * m->width * m->height;
                if (tmpmax < ft)   tmpmax = ft;
                if (tmpmax < need) tmpmax = need;
            }
        }
        net->cnnbufsize = tmpmax;
        net->cnntempbuf = tmpmax ? malloc(tmpmax * sizeof(float)) : NULL;
        net->layerbuf   = total  ? malloc(total  * sizeof(float)) : NULL;
        if (!net->weight_buf || !mat->data ||
            (total  && !net->layerbuf) ||
            (tmpmax && !net->cnntempbuf)) { printf("failed to allocate buffers for net_load !\n"); net_free(net); net = NULL; return net; }
        /* 3) 给每个槽位切 arena 切片，并一次性填好 pad 边缘：
         *    写方永远只写内部区域，边缘值跨帧有效，无需每帧重填。
         *    maxpool 的输入边缘填 -FLT_MAX，使取 max 时自动忽略边缘。 */
        for (i=1; i<=layers; i++) {
            MATRIX *m = &net->layer_list[i].matrix;
            size = (m->width + m->pad * 2) * (m->height + m->pad * 2) * m->channels;
            m->data = net->layerbuf + off;
            off += size;
            matrix_fill_pad(m, net->layer_list[i].type == LAYER_TYPE_MAXPOOL ? -FLT_MAX : 0);
        }
    }
    return net;
}

/*
 * net_bn_fold - 把 batchnorm 折叠进卷积权重副本（优化路径专用，只写 data2/bias2）
 *
 * 原式: out = act( (dot - mean) * invvar * scale + bias )
 * 折叠: 令 a_j = scale_j / sqrt(var_j + eps)，则
 *          W'_jk = W_jk * a_j,   b'_j = bias_j - mean_j * a_j,
 *          out = act( dot' + b' ),  dot' 用 W' 计算。
 * 内循环因此不再有 BN 分支/减法/乘法。注意此处使用精确 sqrtf 而非原始路径的
 * fast_inverse_sqrt，数值更准，但与原始路径存在正常浮点差异（容差比对）。
 * 只读 data/bias/scale/mean/var，重复调用安全；bn_fused 置位后不再重复折叠。
 */
void net_bn_fold(NET *net)
{
    int i, j, k;
    if (!net || net->bn_fused) return;
    for (i=0; i<net->layer_num; i++) {
        LAYER  *l = net->layer_list + i;
        FILTER *f = &l->filter;
        int     ft;
        if (l->type != LAYER_TYPE_CONV || !f->batchnorm) continue;
        ft = f->size * f->size * f->channels;
        for (j=0; j<f->n; j++) {
            float a = f->scale[j] / sqrtf(f->rolling_variance[j] + 0.00001f);
            f->bias2[j] = f->bias[j] - f->rolling_mean[j] * a;
            for (k=0; k<ft; k++) f->data2[j * ft + k] = f->data[j * ft + k] * a;
        }
    }
    net->bn_fused = 1;
}
