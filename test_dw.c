/*
 * test_dw.c - conv_depthwise 单元测试
 *
 * 编译: gcc -O2 -I. test_dw.c kernels/common/kernel_shared.c -lm -o test_dw && ./test_dw
 * 说明: 直接 #include kernels/opt/kernel_conv_opt.c 以访问其中的 static 内核函数。
 *       对每组 (W,H,C,size,stride,输出pad) 形状，用伪随机输入/权重分别跑
 *       conv_depthwise 与朴素参考实现，逐元素比较（阈值 1e-4）。
 *
 * 重要教训（写参考实现时必须遵守）：窗口锚点使用 padded 数组坐标——
 * 输出像素 (x,y) 的窗口 = padded[y*STRIDE .. y*STRIDE+SIZE-1] 行、
 * x 同理，不加 +PAD 偏移。曾因参考实现与内核犯同样的 +PAD 错误而漏测。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernels/opt/kernel_conv_opt.c"

static unsigned seed = 12345;
static float rnd(void) { seed = seed * 1103515245u + 12345u; return ((seed >> 16) % 2000) / 1000.0f - 1.0f; }

static int run_case(int W, int H, int C, int SIZE, int STRIDE, int OPAD)
{
    int PAD = 1;
    int mwi = W + PAD * 2, mhi = H + PAD * 2;
    int ow = (W - SIZE + PAD * 2) / STRIDE + 1, oh = (H - SIZE + PAD * 2) / STRIDE + 1;
    int mwo = ow + OPAD * 2, mho = oh + OPAD * 2;
    float *in  = calloc(mwi * mhi * C, sizeof(float));
    float *out = calloc(mwo * mho * C, sizeof(float));
    float *ref = calloc(ow * oh * C, sizeof(float));
    float *Wt  = malloc(SIZE * SIZE * C * sizeof(float));
    float *B   = malloc(C * sizeof(float));
    int c, x, y, ky, kx, i, bad = 0;

    /* 随机填满整个 padded 缓冲，再显式清零边缘区，模拟引擎的 margin 约定 */
    for (i = 0; i < mwi * mhi * C; i++) in[i] = rnd();
    for (c = 0; c < C; c++)
        for (y = 0; y < mhi; y++)
            for (x = 0; x < mwi; x++)
                if (x < PAD || x >= mwi - PAD || y < PAD || y >= mhi - PAD)
                    in[c * mwi * mhi + y * mwi + x] = 0.0f;
    for (i = 0; i < SIZE * SIZE * C; i++) Wt[i] = rnd();
    for (c = 0; c < C; c++) B[c] = rnd();

    {
        MATRIX mi = { W, H, C, PAD, in };
        MATRIX mo = { ow, oh, C, OPAD, out };
        conv_depthwise(&mi, &mo, Wt, B, SIZE, STRIDE, ACTIVATE_TYPE_LINEAR);
    }
    /* 朴素参考实现：与 im2row 相同的 padded 坐标锚点 */
    for (c = 0; c < C; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                float acc = B[c];
                for (ky = 0; ky < SIZE; ky++)
                    for (kx = 0; kx < SIZE; kx++)
                        acc += Wt[c * SIZE * SIZE + ky * SIZE + kx] *
                               in[c * mwi * mhi + (y * STRIDE + ky) * mwi + (x * STRIDE + kx)];
                ref[c * ow * oh + y * ow + x] = acc;
            }
    /* 从含 OPAD 边缘的输出缓冲中取内部区域逐元素比对 */
    for (c = 0; c < C; c++)
        for (y = 0; y < oh; y++)
            for (x = 0; x < ow; x++) {
                float g = out[c * mwo * mho + (OPAD + y) * mwo + OPAD + x];
                float r = ref[c * ow * oh + y * ow + x];
                if (fabsf(g - r) > 1e-4f) {
                    if (bad < 5) printf("   bad at c=%d y=%d x=%d got=%f want=%f\n", c, y, x, g, r);
                    bad++;
                }
            }
    free(in); free(out); free(ref); free(Wt); free(B);
    return bad;
}

int main(void)
{
    /* 覆盖：方阵/非方阵、stride 1/2、输出 pad 有无、大通道数(depthwise 224)、奇数尺寸 */
    int cases[][6] = {
        {16, 16, 8, 3, 1, 0},
        {16, 16, 8, 3, 1, 1},
        {16, 16, 8, 3, 2, 0},
        {16, 16, 8, 3, 2, 1},
        {160, 160, 8, 3, 1, 1},
        {160, 160, 8, 3, 2, 1},
        {10, 10, 224, 3, 1, 0},
        {7, 9, 5, 3, 2, 2},
        {20, 12, 32, 3, 2, 1},
    };
    int i, fails = 0;
    for (i = 0; i < 9; i++) {
        int bad = run_case(cases[i][0], cases[i][1], cases[i][2], cases[i][3], cases[i][4], cases[i][5]);
        printf("case %d: W=%d H=%d C=%d S=%d st=%d opad=%d -> bad=%d\n",
               i, cases[i][0], cases[i][1], cases[i][2], cases[i][3], cases[i][4], cases[i][5], bad);
        fails += bad;
    }
    printf(fails ? "DW TEST FAIL\n" : "DW TEST ALL PASS\n");
    return 0;
}
