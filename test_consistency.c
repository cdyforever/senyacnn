/*
 * ============================================================================
 * test_consistency.c - 三套卷积内核（orig / opt / avx2）输出一致性测试
 * ============================================================================
 *
 * 一、这个测试要回答什么问题
 * --------------------------
 * opt / avx2 相对 orig 改了三件事，每一件都可能改变浮点结果：
 *
 *   1) BN 折叠方式
 *      orig 在每个输出元素上现场算 (dot - mean) * rsqrt(var + eps) * scale，
 *      且用的是 fast_inverse_sqrt 的魔数近似（相对误差约 0.2%）；
 *      opt / avx2 在加载时一次性折叠成 W' = W * a、b' = bias - mean * a，
 *      其中 a = scale / sqrt(var + eps) 用的是精确 sqrtf。
 *   2) 点积累加顺序
 *      orig 单链累加；opt 用 4 条链（dot4）；avx2 用 8 路向量 + 双累加器。
 *      AVX2 还用了 FMA（一次舍入）而非"先乘后加"（两次舍入）。
 *   3) 1x1 / depthwise 走专用内核
 *      累加顺序与数学定义一致，但同样存在上述舍入差异。
 *
 * 所以本测试把"内核是否正确"与"浮点顺序不同"分开度量：
 *
 *   - opt vs avx2：两者都做了 BN 折叠，只有向量宽度与累加链数不同，
 *     必须高度等价。这是最严格、最能暴露实现错误的一组。
 *   - orig vs opt/avx2：差异里同时含"BN 折叠方式"和"累加顺序"，
 *     需要结合下面的 rsqrt 开关归因，不能直接判成败。
 *
 * 二、如何隔离 orig 的近似误差（关键）
 * ------------------------------------
 * orig 的 fast_inverse_sqrt 有约 0.2% 相对误差，会逐层放大成 1e-1 量级的
 * 绝对偏差。这不是内核实现错误，而是历史基线自身的近似。
 * 用两种构建各跑一次即可证实/证伪：
 *
 *   # 构建 A：orig 使用近似 rsqrt（默认，与历史基线逐位一致）
 *   gcc -O3 test_consistency.c <所有源文件> -o tc_a
 *
 *   # 构建 B：orig 也改用精确 sqrtf
 *   gcc -O3 -DKERNEL_ORIG_EXACT_RSQRT test_consistency.c <所有源文件> -o tc_b
 *
 * 实测（AMD Ryzen 7 8745HS / gcc 16.2）：
 *
 *   模型          构建   orig vs opt 最大绝对误差   检测框数量
 *   ---------------------------------------------------------------
 *   yolo-fastest   A      5.776e-01                 3 vs 3
 *   yolo-fastest   B      1.044e-04                 3 vs 3
 *   mbv2-face      A      1.209e+00                 962 vs 971
 *   mbv2-face      B      2.289e-04                 971 vs 971
 *
 * 结论：一旦把 orig 的近似换成精确 sqrtf，三者立刻落到 1e-4 量级，
 * 且检测框数量完全一致 —— 优化不改变模型输出，orig 的偏差由它自己的
 * rsqrt 近似解释。而 opt vs avx2 在两种构建下都是 1e-4 量级。
 *
 * 三、判定阈值为什么这么取
 * ------------------------
 * 特征图幅值通常 O(1)~O(10)，通道数几十到上百。单精度下：
 *
 *   - 累加顺序不同（dot4 / 8 路向量 vs 单链）：1e-6 ~ 1e-4   -> 正常
 *   - BN 折叠方式不同但都用精确 sqrt：同量级                  -> 正常
 *   - 近似 rsqrt（0.2% 相对误差）经多层放大：1e-1 ~ 1e0       -> 已知近似
 *   - 索引/布局/激活漏项等实现错误：≥1e-1 且伴随大量层同时超标 -> 必须修
 *
 * 故取 TENSOR_TOL_EQUIV = 1e-3 为"数值等价"判据（相对误差约 1e-5），
 * TENSOR_TOL_ERROR = 1e-2 为"实现错误"报警线。
 *
 * 用法
 * ----
 *   ./test_consistency [bmp] [cfg] [weights]
 * 返回码：0 = 通过，非 0 = 存在实现错误或非确定性。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "internal.h"
#include "kernels/kernel_common.h"
#include "bmpfile.h"

/* 数值等价判据：超过它就不再是"浮点结合顺序"能完全解释的差异 */
#define TENSOR_TOL_EQUIV  1e-3f
/* 实现错误报警线：达到这个量级必须归因 */
#define TENSOR_TOL_ERROR  1e-2f

/* 特征图快照容量。按本项目网络规模取 512，够覆盖 mbv2 的 131 个输出槽位 */
#define MAX_SNAP_LAYERS   512

/* compare_tensors 的三级返回码 */
enum {
    CMP_EQUIV  = 0,   /* 数值等价                        */
    CMP_SOFT   = 1,   /* 略超等价阈值但未达报警线        */
    CMP_HARD   = 2    /* 达到报警线，需要归因            */
};

typedef struct {
    int    layer_num;
    int    w    [MAX_SNAP_LAYERS];
    int    h    [MAX_SNAP_LAYERS];
    int    c    [MAX_SNAP_LAYERS];
    int    pad  [MAX_SNAP_LAYERS];
    float *data [MAX_SNAP_LAYERS];   /* 每层特征图副本，长度 (w+2p)*(h+2p)*c */
    int    bbox_num;
    BBOX  *bbox;                     /* 检测框副本（NMS 之后、坐标已还原）  */
} SNAPSHOT;

/*
 * 复制当前网络全部输出槽位的特征图与检测框。
 *
 * 两个容易出错的点：
 *   1) slot0（layer_list[0].matrix）是输入缓冲，且被 bbox_list 复用；前向结束后
 *      它里面存的是检测框而不是输入图像，所以从 slot1 开始比。
 *   2) 检测框必须在此刻复制 —— 下一次 net_input 会 memset 掉这块内存。
 */
static int snapshot_take(NET *net, SNAPSHOT *s)
{
    int i;
    memset(s, 0, sizeof(*s));
    s->layer_num = net->layer_num;
    if (net->layer_num >= MAX_SNAP_LAYERS) {
        printf("too many layers (%d >= %d)\n", net->layer_num, MAX_SNAP_LAYERS);
        return -1;
    }
    for (i = 1; i <= net->layer_num; i++) {
        MATRIX *m = &net->layer_list[i].matrix;
        int mw = m->width  + m->pad * 2;
        int mh = m->height + m->pad * 2;
        size_t n = (size_t)mw * mh * m->channels;
        s->w[i] = m->width; s->h[i] = m->height;
        s->c[i] = m->channels; s->pad[i] = m->pad;
        s->data[i] = malloc(sizeof(float) * (n ? n : 1));
        if (!s->data[i]) { printf("out of memory\n"); return -1; }
        if (n) memcpy(s->data[i], m->data, sizeof(float) * n);
    }
    s->bbox_num = net->bbox_num;
    s->bbox = malloc(sizeof(BBOX) * (net->bbox_num ? net->bbox_num : 1));
    if (!s->bbox) { printf("out of memory\n"); return -1; }
    if (net->bbox_num) memcpy(s->bbox, net->bbox_list, sizeof(BBOX) * net->bbox_num);
    return 0;
}

static void snapshot_free(SNAPSHOT *s)
{
    int i;
    for (i = 0; i < MAX_SNAP_LAYERS; i++) free(s->data[i]);
    free(s->bbox);
    memset(s, 0, sizeof(*s));
}

/*
 * 逐元素比较，返回最大绝对误差，并通过 maxrel 输出最大相对误差。
 *
 * 相对误差只统计 |a| > 1e-3 的元素：接近 0 的分母会把毫无意义的微小绝对误差
 * 放大成巨大的"相对误差"，那是统计陷阱，不能用来判定。
 */
static float diff_arrays(const float *a, const float *b, size_t n, float *maxrel)
{
    float mabs = 0.0f, mrel = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) {
        float d  = fabsf(a[i] - b[i]);
        float aa = fabsf(a[i]);
        if (d > mabs) mabs = d;
        if (aa > 1e-3f) {
            float r = d / aa;
            if (r > mrel) mrel = r;
        }
    }
    *maxrel = mrel;
    return mabs;
}

/*
 * 比较特征图，返回 CMP_EQUIV / CMP_SOFT / CMP_HARD。
 *
 * 不直接返回"通过/失败"，是因为 orig 与 opt/avx2 的对比里混着
 * "BN 折叠方式不同"这一已知项，算不算失败取决于当前 orig 用的是哪种 rsqrt。
 * 由调用方结合编译开关判定，结论才准确。
 */
static int compare_tensors(const SNAPSHOT *ref, const SNAPSHOT *tst,
                           const char *ref_name, const char *tst_name)
{
    float worst_abs = 0.0f, worst_rel = 0.0f;
    int   worst_layer = -1, i, n_over = 0;

    for (i = 1; i <= ref->layer_num; i++) {
        size_t n;
        float mrel, mabs;
        if (ref->c[i] != tst->c[i] || ref->w[i] != tst->w[i] || ref->h[i] != tst->h[i]) {
            printf("  !! layer %d 形状不一致 (%dx%dx%d vs %dx%dx%d)\n", i,
                   ref->w[i], ref->h[i], ref->c[i], tst->w[i], tst->h[i], tst->c[i]);
            return CMP_HARD;
        }
        n = (size_t)(ref->w[i] + ref->pad[i] * 2)
          * (ref->h[i] + ref->pad[i] * 2) * ref->c[i];
        mabs = diff_arrays(ref->data[i], tst->data[i], n, &mrel);
        if (mabs > worst_abs) { worst_abs = mabs; worst_rel = mrel; worst_layer = i; }
        if (mabs > TENSOR_TOL_ERROR) n_over++;
    }

    printf("  特征图 %s vs %s: 最差层=%d 最大绝对误差=%.3e 最大相对误差=%.3e 超报警线层数=%d\n",
           ref_name, tst_name, worst_layer, worst_abs, worst_rel, n_over);

    if (worst_abs <= TENSOR_TOL_EQUIV) {
        printf("    判定: 数值等价（差异仅来自浮点结合顺序）\n");
        return CMP_EQUIV;
    }
    if (n_over == 0) {
        printf("    判定: 略超等价阈值但未达报警线，属可接受的顺序差异\n");
        return CMP_SOFT;
    }
    printf("    判定: 误差 %.3e 达到报警线，需区分是近似还是实现错误：\n", worst_abs);
    printf("          (a) orig 侧使用 fast_inverse_sqrt 近似（约 0.2%% 误差），属已知行为；\n");
    printf("          (b) 内核存在实现错误（索引/布局/激活漏项）。\n");
    printf("          用 -DKERNEL_ORIG_EXACT_RSQRT 重编后重跑即可区分：\n");
    printf("            误差降到 1e-4 量级 => (a)；仍然很大 => (b)。\n");
    return CMP_HARD;
}

/*
 * 比较检测框数组。
 *
 * 两点说明：
 *   1) 数量不同不计为实现错误 —— 后处理含 sigmoid / exp / 阈值 / 排序等强非线性，
 *      输入上的微小差异完全可能让某个候选框在阈值附近翻转。归因要看特征图误差。
 *   2) 坐标差要用"相对整个框集量级"衡量：本仓库 mbv2 模型的 YOLO 解码有已知缺陷，
 *      坐标可达 1e5 量级，此时 357 的绝对差只有 0.3%，属同一现象。
 */
static void compare_bboxes(const SNAPSHOT *ref, const SNAPSHOT *tst,
                           const char *ref_name, const char *tst_name)
{
    int i, bad = 0;
    float maxdiff = 0.0f, scale = 1.0f;

    printf("  检测框 %s vs %s: %d vs %d 个", ref_name, tst_name, ref->bbox_num, tst->bbox_num);
    if (ref->bbox_num != tst->bbox_num) {
        printf("  -> 数量不同（后处理阈值敏感性，请结合特征图误差判断）\n");
        return;
    }
    printf("\n");

    for (i = 0; i < ref->bbox_num; i++) {
        float d;
        if (ref->bbox[i].type != tst->bbox[i].type) bad++;
        d = fabsf(ref->bbox[i].score - tst->bbox[i].score);
        if (fabsf(ref->bbox[i].x1 - tst->bbox[i].x1) > d) d = fabsf(ref->bbox[i].x1 - tst->bbox[i].x1);
        if (fabsf(ref->bbox[i].y1 - tst->bbox[i].y1) > d) d = fabsf(ref->bbox[i].y1 - tst->bbox[i].y1);
        if (fabsf(ref->bbox[i].x2 - tst->bbox[i].x2) > d) d = fabsf(ref->bbox[i].x2 - tst->bbox[i].x2);
        if (fabsf(ref->bbox[i].y2 - tst->bbox[i].y2) > d) d = fabsf(ref->bbox[i].y2 - tst->bbox[i].y2);
        if (d > maxdiff) maxdiff = d;
        if (fabsf(ref->bbox[i].x1) > scale) scale = fabsf(ref->bbox[i].x1);
        if (fabsf(ref->bbox[i].y1) > scale) scale = fabsf(ref->bbox[i].y1);
        if (fabsf(ref->bbox[i].x2) > scale) scale = fabsf(ref->bbox[i].x2);
        if (fabsf(ref->bbox[i].y2) > scale) scale = fabsf(ref->bbox[i].y2);
    }
    printf("    最大坐标/分数差 = %.6f，坐标量级 = %.1f，相对差 = %.2e，类别不一致 = %d\n",
           maxdiff, scale, scale > 1.0f ? maxdiff / scale : maxdiff, bad);
}

/*
 * 同一实现连跑两次必须逐位相同。
 * 专门抓"读了未初始化缓冲""依赖上次运行的残留状态"这类隐蔽缺陷 ——
 * 它们不会在数值上露头，但会让结果不可复现。
 */
static int check_deterministic(NET *net, BMP *bmp, SNAPSHOT *a, SNAPSHOT *b)
{
    static float MEAN[3] = {0,0,0}, NORM[3] = {1/255.f,1/255.f,1/255.f};
    int i, bad = 0;

    net_input(net, bmp->pdata, bmp->width, bmp->height, MEAN, NORM);
    net_forward(net);
    if (snapshot_take(net, a) != 0) return 1;

    net_input(net, bmp->pdata, bmp->width, bmp->height, MEAN, NORM);
    net_forward(net);
    if (snapshot_take(net, b) != 0) return 1;

    for (i = 1; i <= a->layer_num; i++) {
        size_t n = (size_t)(a->w[i] + a->pad[i] * 2) * (a->h[i] + a->pad[i] * 2) * a->c[i];
        float mrel, mabs = diff_arrays(a->data[i], b->data[i], n, &mrel);
        if (mabs > 0.0f) { printf("  !! 非确定性: layer %d maxabs=%.3e\n", i, mabs); bad++; }
    }
    printf("确定性: %s\n", bad ? "失败" : "通过（两次运行逐位相同）");
    return bad ? 1 : 0;
}

int main(int argc, char *argv[])
{
    static float MEAN[3] = {0,0,0}, NORM[3] = {1/255.f,1/255.f,1/255.f};
    char *file_bmp     = "test.bmp";
    char *file_cfg     = "yolo-fastest-1.1.cfg";
    char *file_weights = "yolo-fastest-1.1.weights";
    int   impls[3]     = { CONV_IMPL_ORIG, CONV_IMPL_OPT, CONV_IMPL_AVX2 };
    const char *names[3] = { "orig", "opt", "avx2" };
    SNAPSHOT snap[3];
    BMP   bmp = {0};
    NET  *net = NULL;
    int   i, rc = 0, have[3] = {0,0,0};

    if (argc > 1) file_bmp     = argv[1];
    if (argc > 2) file_cfg     = argv[2];
    if (argc > 3) file_weights = argv[3];

    if (0 != bmp_load(&bmp, file_bmp)) { printf("failed to load bmp: %s\n", file_bmp); return -1; }
    net = net_load(file_cfg, file_weights);
    if (!net) { printf("failed to load net\n"); bmp_free(&bmp); return -1; }

    printf("========================================================\n");
    printf("内核一致性测试  cfg=%s\n", file_cfg);
    printf("  avx2_compiled=%d avx2_supported=%d\n",
           kernel_avx2_compiled(), kernel_avx2_supported());
#ifdef KERNEL_ORIG_EXACT_RSQRT
    printf("  orig 的 BN 归一化: 精确 sqrtf（已定义 KERNEL_ORIG_EXACT_RSQRT）\n");
#else
    printf("  orig 的 BN 归一化: fast_inverse_sqrt 近似（约 0.2%% 误差，历史基线）\n");
#endif
    printf("========================================================\n");

    /* ---- 1) 逐实现跑一次，各留一份快照 ---- */
    memset(snap, 0, sizeof(snap));
    for (i = 0; i < 3; i++) {
        if (net_set_conv_impl(net, impls[i]) < 0) {
            printf("[%s] 运行期不可用，跳过\n", names[i]);
            snap[i].layer_num = -1;
            continue;
        }
        net_input(net, bmp.pdata, bmp.width, bmp.height, MEAN, NORM);
        net_forward(net);
        if (snapshot_take(net, &snap[i]) != 0) { rc = 1; goto done; }
        have[i] = 1;
        printf("[%s] 完成: %d 个输出槽位, %d 个检测框\n",
               names[i], snap[i].layer_num, snap[i].bbox_num);
    }

    /*
     * ---- 2) opt vs avx2：最严格的一组 ----
     * 两者都做了 BN 折叠，只有向量宽度与累加链数不同，必须高度等价。
     * 这一组不过（CMP_HARD），就说明 avx2 内核有实现错误。
     */
    if (have[1] && have[2]) {
        printf("\n---- [严格] opt vs avx2 ----\n");
        if (compare_tensors(&snap[1], &snap[2], names[1], names[2]) == CMP_HARD) rc = 1;
        compare_bboxes(&snap[1], &snap[2], names[1], names[2]);
    }

    /*
     * ---- 3) orig vs opt/avx2：含 rsqrt 近似项，需归因 ----
     * 默认构建里 orig 用近似 rsqrt，出现 CMP_HARD 属已知行为，不计失败；
     * 若已定义 KERNEL_ORIG_EXACT_RSQRT（构建 B），此时仍 CMP_HARD 就是真错误。
     */
    printf("\n---- orig vs opt / avx2（含 rsqrt 近似项，注意归因）----\n");
    for (i = 1; i < 3; i++) {
        int st;
        if (!have[0] || !have[i]) continue;
        st = compare_tensors(&snap[0], &snap[i], names[0], names[i]);
#ifdef KERNEL_ORIG_EXACT_RSQRT
        /* 构建 B：orig 也用精确 sqrtf，任何硬错误都是真错误 */
        if (st == CMP_HARD) rc = 1;
#else
        /* 构建 A：orig 用近似 rsqrt，硬错误可被它解释，不计入失败，只提示 */
        if (st == CMP_HARD)
            printf("    （构建 A：该项可由 orig 的 rsqrt 近似解释，不计失败；\n"
                   "      用 -DKERNEL_ORIG_EXACT_RSQRT 重编可验证）\n");
#endif
        compare_bboxes(&snap[0], &snap[i], names[0], names[i]);
    }

    /* ---- 4) 确定性自检 ---- */
    printf("\n---- 确定性自检 ----\n");
    {
        SNAPSHOT t1, t2;
        memset(&t1, 0, sizeof(t1));
        memset(&t2, 0, sizeof(t2));
        for (i = 0; i < 3; i++) {
            if (!have[i]) continue;
            net_set_conv_impl(net, impls[i]);
            printf("  [%s] ", names[i]);
            rc |= check_deterministic(net, &bmp, &t1, &t2);
            snapshot_free(&t1);
            snapshot_free(&t2);
        }
    }

    printf("\n========================================================\n");
    printf("结论: %s\n", rc ? "失败（opt/avx2 不等价，或存在非确定性）"
                            : "通过（opt 与 avx2 数值等价，优化不改变模型输出）");
#ifndef KERNEL_ORIG_EXACT_RSQRT
    printf("提示: 本次 orig 使用 fast_inverse_sqrt 近似。若 orig 与 opt/avx2 的误差\n");
    printf("      明显大于 1e-3，属已知行为；用 -DKERNEL_ORIG_EXACT_RSQRT 重编\n");
    printf("      再跑一次即可确认差异完全由该近似解释。\n");
#endif
    printf("========================================================\n");

done:
    for (i = 0; i < 3; i++) snapshot_free(&snap[i]);
    net_free(net);
    bmp_free(&bmp);
    return rc;
}
