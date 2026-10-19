/*
 * test_1x1.c - conv_1x1 单元测试
 *
 * 编译: gcc -O2 -I. test_1x1.c -lm -o test_1x1 && ./test_1x1
 * 直接 #include kernels/opt/kernel_conv_opt.c 以访问 static 内核函数。
 * 特意让输出槽位带 pad(OPAD=1)，验证内核对"输出 pad 属于下一层"
 * 这一布局约定的正确处理。线性激活下逐元素与朴素三重循环比对。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernels/opt/kernel_conv_opt.c"

int main(void)
{
    int W = 16, H = 16, C = 8, N = 4, OPAD = 1;
    int plane = W * H;
    float *in  = malloc(plane * C * sizeof(float));
    float *out = calloc((W + OPAD * 2) * (H + OPAD * 2) * N, sizeof(float));
    float *Wt  = malloc(N * C * sizeof(float));
    float *B   = malloc(N * sizeof(float));
    unsigned seed = 777;
    int i, c, x, y;
    float maxdiff = 0;

    for (i = 0; i < plane * C; i++) { seed = seed * 1103515245u + 12345u; in[i] = ((seed >> 16) % 2000) / 1000.0f - 1.0f; }
    for (i = 0; i < N * C; i++)      { seed = seed * 1103515245u + 12345u; Wt[i] = ((seed >> 16) % 1000) / 500.0f - 1.0f; }
    for (i = 0; i < N; i++)          { seed = seed * 1103515245u + 12345u; B[i]  = ((seed >> 16) % 1000) / 500.0f - 1.0f; }

    {
        MATRIX mi = { W, H, C, 0, in };          /* 1x1 内核要求输入 pad==0 */
        MATRIX mo = { W, H, N, OPAD, out };
        conv_1x1(&mi, &mo, Wt, B, ACTIVATE_TYPE_LINEAR);
    }

    /* 参考实现：直接按定义求和，从含 pad 的输出缓冲取内部元素比对 */
    for (i = 0; i < N; i++)
        for (y = 0; y < H; y++)
            for (x = 0; x < W; x++) {
                float acc = B[i];
                for (c = 0; c < C; c++) acc += Wt[i * C + c] * in[c * plane + y * W + x];
                {
                    int mwo = W + OPAD * 2;
                    float got = out[i * (mwo * (H + OPAD * 2)) + OPAD * mwo + OPAD + y * mwo + x];
                    float d = fabsf(got - acc);
                    if (d > maxdiff) maxdiff = d;
                }
            }
    printf("conv_1x1 maxdiff=%f\n", maxdiff);
    return 0;
}
