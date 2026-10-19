# 卷积内核：三套实现剖析

> 本文逐个拆解 `kernels/{orig,opt,avx2}` 三套卷积实现：算法、边界推导、为什么这么做，
> 以及**哪些优化试过但没用**。如果你想自己加第四套（NEON、SVE、RVV），看[最后一节](#七如何新增第四套内核)。

---

## 目录

- [一、三套实现的共同契约](#一三套实现的共同契约)
- [二、orig：原始标量实现](#二orig原始标量实现)
- [三、opt：优化标量实现](#三opt优化标量实现)
- [四、avx2：AVX2 + FMA 实现](#四avx2avx2--fma-实现)
- [五、三套实现横向对照](#五三套实现横向对照)
- [六、试过但无效的优化](#六试过但无效的优化)
- [七、如何新增第四套内核](#七如何新增第四套内核)
- [八、写内核的检查清单](#八写内核的检查清单)

---

## 一、三套实现的共同契约

三套内核的函数签名完全一致：

```c
void kernel_conv_forward_XXX(NET *net, LAYER *ilayer, LAYER *olayer);
```

它们共享以下前提（细节见 [architecture.md](architecture.md)）：

| 契约 | 内容 |
|---|---|
| 数据布局 | CHW；平面实际尺寸 `(w+2p) × (h+2p)`；内部像素 `(x,y)` 地址 = `data + y*mw + x` |
| 窗口锚点 | `padded[oy*stride + fy][ox*stride + fx]`，**不加 `+pad`** |
| 写范围 | **只写输出平面的内部区域**，绝不碰 pad 边缘 |
| 权重 | `orig` 用 `data`/`bias`/`scale`/`mean`/`var`；`opt`/`avx2` 用折叠后的 `data2`/`bias2` |
| 激活 | 内层用 `kernel_actv()`（relu/leaky 展开为无分支表达式），其余转 `activate()` |
| 无堆操作 | 临时缓冲来自 `net->cnntempbuf`，不在此处 `malloc`（兜底分支除外） |

`filter.channels` 在 `parser.c` 里已经被算成**每组的输入通道数**，
所以 depthwise 的判据是 `filter.channels == 1`。

---

## 二、orig：原始标量实现

### 2.1 算法

```
for n in 0..n-1:                      # 输出通道
  for oy, ox:                         # 输出像素（整平面跳跃）
    im2row(窗口) -> buf[ftotal]       # 每个输出像素都重做一次拷贝
    acc = 0                           # 单链累加
    for j in 0..ftotal-1:
        acc += data[n][j] * buf[j]
    # 每元素现场算 BN（含近似开方）
    acc = (acc - mean[n]) * fast_inverse_sqrt(var[n] + 1e-5) * scale[n] + bias[n]
    out[n][oy][ox] = activate(acc)
```

### 2.2 三处性能短板

| # | 问题 | 代价 |
|---|---|---|
| 1 | **每个输出元素一次 `fast_inverse_sqrt`** | 魔数 + 一次牛顿迭代 ≈ 10 条指令，对 `ftotal=9` 的 layer 快跟点积本身一样贵 |
| 2 | **每个输出像素一次 im2row** | im2col 拷贝量 = 权重矩阵大小 × 输出像素数，与乘加同量级；且同一份输入被 `n` 个滤波器重复拷贝 `n` 次 |
| 3 | **整平面跳跃写** | `out` 的写指针在每个 `(oy,ox)` 上跳 `mw*mh` 个 float，每次都落在新的 cache 行 |

### 2.3 为什么仍然保留它

| 理由 | 说明 |
|---|---|
| **逐位可复现的历史基线** | 它保留了 `fast_inverse_sqrt` 的近似与单链累加顺序，对 `yolo-fastest-1.1` 的输出与改造前的版本逐位相同。任何"结果变了"的疑问都可以拿它当参照 |
| **独立实现 = 独立证据** | 它和 `opt` 在算法结构上差得足够远，两者一致就基本排除了共模错误 |
| **性能下限参照** | 加速比的分母 |

### 2.4 `KERNEL_ORIG_EXACT_RSQRT` 开关

```c
#ifdef KERNEL_ORIG_EXACT_RSQRT
  #define ORIG_INV_SQRT(x) (1.0f / sqrtf(x))     /* 精确 */
#else
  #define ORIG_INV_SQRT(x) kernel_fast_inverse_sqrt(x)   /* 魔数近似，默认 */
#endif
```

**用途**：把"`orig` 的近似误差"与"`opt` 的实现错误"分离。
默认构建保持历史行为；加 `-DKERNEL_ORIG_EXACT_RSQRT` 重编后，
`orig` 与 `opt` 的最大偏差从 **1e-1 量级**掉到 **1e-4 量级**，
证明优化本身没有引入错误。详见 [testing.md](testing.md)。

---

## 三、opt：优化标量实现

纯标量，不依赖任何 CPU 扩展。相对 `orig` 做了四项改动。

### 3.1 改动一：BN 折叠到权重

见 [architecture.md 第八节](architecture.md#八bn-折叠)。效果：内层循环里彻底消失 BN 运算，
顺带去掉每个元素一次的近似开方。

### 3.2 改动二：按几何分派到专用内核

```c
is_dw  = (filter.channels == 1 && filter.groups == mati.channels && filter.groups >= 1);
is_1x1 = (filter.size == 1 && filter.stride == 1 && mati.pad == 0 && filter.groups == 1);

if (is_dw)        -> conv_depthwise()          # 直卷积，完全跳过 im2col
else if (is_1x1)  -> conv_1x1()                # 通道点积，跳过 im2col
else              -> 通用 im2col + dot4
```

**为什么值得**：MobileNet 系网络里 1×1 与 depthwise 占了绝大多数层，
而通用 im2col 对它们的拷贝量与计算量相当 —— 一半时间花在搬数据上。

| 形状 | 通用 im2col 的浪费 |
|---|---|
| 1×1, stride 1, C→N | `ftotal = C`；im2col 拷贝 `C` 个 float 只为算一次点积，还要 `ow*oh` 份 |
| depthwise 3×3 | `n == 1` 每组只有一个滤波器，**im2col 的拷贝完全无法被摊薄** —— 拷一次只算一次 |

### 3.3 改动三：通用路径的一次性 im2col

`orig` 是"每输出像素拷贝一次，然后 `n` 个滤波器各自重读"。`opt` 改成：

```
for n in 0..groups-1:                # 按组
    im2col 整组 -> col[ftotal][ow*oh]   # 只做一次
    for i in 0..(filter.n/groups)-1:    # 组内所有滤波器复用 col
        for y, x:  out = dot4(W[i], col[y*ow+x]) + b[i]
```

im2col 的拷贝成本被 `filter.n` 摊薄。代价是 `cnntempbuf` 需要 `ftotal × ow × oh` 个 float
（mbv2 实测峰值约 **5.9 MB**），`net_load` 已按全网络最大需求一次分配。

### 3.4 改动四：写顺序改为按行连续

`orig` 的写指针在每个输出像素上跳遍所有通道平面；`opt` 的目标循环是
**滤波器主序**，一个滤波器连续写完整平面：

```
for y:
    for x:  *datao++ = act(dot4(...) + bias)
    datao += mato.pad * 2;       # 跳过右侧 pad 列
```

这带来两个好处：写指针严格递增（不会 miss 同一个 cache 行的后续部分），
且 `dot4` 的输入 `col + (y*ow+x)*ftotal` 也是顺序推进的。

### 3.5 `dot4`：4 路累加器

```c
static float dot4(const float *w, const float *r, int n)
{
    float a0=0, a1=0, a2=0, a3=0;
    for (j = 0; j + 4 <= n; j += 4) {
        a0 += w[j+0]*r[j+0];
        a1 += w[j+1]*r[j+1];
        a2 += w[j+2]*r[j+2];
        a3 += w[j+3]*r[j+3];
    }
    for (; j < n; j++) a0 += w[j]*r[j];
    return (a0+a1) + (a2+a3);
}
```

**为什么需要**：单链 `acc += w[j]*r[j]` 每次迭代依赖上一次的结果，浮点加法延迟约 4 cycle。
现代 CPU 每周期能发起 2 条 FMA，单链把吞吐压到 1/4。拆成 4 条互不相关的链后
可以同时推进，典型提速 **1.5~2×**。

**为什么是 4 而不是 8**：通用路径的 `ftotal` 常见值是 9、27、144、576 等，
4 路在 27 这种小尺寸上也能吃满 3 轮；8 路会让小 `ftotal` 大量退化到尾部循环。
AVX2 版的 `dot4_avx2` 才用 8 路（名字保留 `dot4` 是历史原因）。

**为什么固定求和顺序**：返回 `((a0+a1)+(a2+a3))` 而不是让编译器重排，
保证同一个二进制在同机器上**逐位确定**，便于回归比对。

### 3.6 `conv_depthwise`：直卷积

```c
for c in 0..C-1:                          # 通道独立
    for oy, ox:                           # 输出像素
        acc = 0
        for fy, fx:                       # 窗口
            acc += w[c][fy][fx] * in[c][iy][ix]
        out[c][oy][ox] = act(acc + b[c])
```

不用 im2col：depthwise 的 `n/groups == 1`，没有"复用 im2col"的机会。

### 3.7 `conv_1x1`：通道点积

```c
for i in 0..n-1:                          # 输出通道
    for y, x:                             # 输出像素
        acc = 0
        for c in 0..C-1:                  # 输入通道（跨平面跳 mw*mh）
            acc += w[i][c] * in[c][y][x]
        out[i][y][x] = act(acc + b[i])
```

内层是对 `c` 的累加，每次跨一个完整平面。因为 `mw*mh` 在循环里是常量，
编译器/CPU 的 stride prefetcher 能识别这个模式。

---

## 四、avx2：AVX2 + FMA 实现

### 4.1 指令集是怎么引入的

```c
#pragma GCC push_options
#pragma GCC target("avx2,fma")
   ... 整个文件的向量内核 ...
#pragma GCC pop_options
```

文件级 `#pragma GCC target`，**不改全局 `-march`**。这样产物在任何 x86-64 上都能加载，
是否真正执行由运行期 `__builtin_cpu_supports("avx2")` 决定（见 [architecture.md 第九节](architecture.md#九内核派发与降级)）。

在非 GCC 或非 x86 平台，`KERNEL_HAVE_AVX2` 为 0，整个文件函数体被 `#if` 排除。

### 4.2 `VACT`：向量化激活

```c
#define VACT(a, act, k01) do {                                  \
    if ((act) == ACTIVATE_TYPE_RELU)                            \
        (a) = _mm256_max_ps((a), _mm256_setzero_ps());          \
    else if ((act) == ACTIVATE_TYPE_LEAKY)                      \
        (a) = _mm256_max_ps((a), _mm256_mul_ps((a), (k01)));    \
} while (0)
```

`leaky` 的技巧值得单独说：

```
leaky(x) = x      if x > 0
         = 0.1x   if x <= 0
```

对 `x > 0`：`0.1x < x`，故 `max(x, 0.1x) = x` ✓
对 `x < 0`：`0.1x > x`，故 `max(x, 0.1x) = 0.1x` ✓

**一条 `max` 指令搞定，不需要 `blendv`、比较掩码或分支。** 同理 relu 就是 `max(x, 0)`。

写成宏而非函数，是为了让编译器把它内联进各内核的循环体，避免函数调用打断向量化。

### 4.3 `conv_depthwise_avx2`：3×3 stride 1

```
                    输出一行 8 个像素（ox .. ox+7）
                    ↓
输入行 r0:  [ox] [ox+1] [ox+2] ... [ox+9]        3 次错位 loadu 覆盖 kx=0,1,2
             ↑w0   ↑w1    ↑w2                     重叠部分由 cache / 合并逻辑吸收
```

```c
for (ox = 0; ox + 8 <= ow; ox += 8) {
    a = fmadd(w0, loadu(r0+ox  ), a);
    a = fmadd(w1, loadu(r0+ox+1), a);
    a = fmadd(w2, loadu(r0+ox+2), a);
    a = fmadd(w3, loadu(r1+ox  ), a);
    ...
    a = fmadd(w8, loadu(r2+ox+2), a);
    storeu(orow + ox, act(a));
}
```

**权重常驻寄存器**：`w0..w8` 用 `_mm256_set1_ps` 广播一次，整个 `ox` 循环期间不重新加载
（编译器把它们放在 ymm 寄存器里，9 个权重占 9 个 ymm，剩 7 个给累加器与 load）。

**边界安全性**（向量化最容易出错的地方）：

```
8 个输出需要的输入列范围 = [ox, ox+9]，共 10 列
循环条件 ox + 8 <= ow  =>  ox <= ow - 8
最大读取列号 = ox + 2 + 7 = ow + 1
而平面行宽 mwi = ow + 2*pad = ow + 2（本内核用于 pad=1 的 3×3）
故 ow + 1 = mwi - 1，正好是行内最后一个合法下标，不越界。
```

尾部不足 8 个的部分用标量循环补齐（`for (; ox < ow; ox++)`）。

### 4.4 `conv_depthwise_s2_avx2`：3×3 stride 2

stride 2 时相邻输出对应输入的**相隔 2 列**，AVX2 没有便宜的 stride-2 load。
`dw_gather2()` 用 3 条指令取出 8 个跨步元素：

```
a = loadu(p + 0)  = [p0 p1 p2 p3 | p4  p5  p6  p7 ]
b = loadu(p + 8)  = [p8 p9 p10 p11| p12 p13 p14 p15]

shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0))
    每 lane 内取 (a[x0], a[x2], b[x0], b[x2]):
    = [p0 p2 p8 p10 | p4 p6 p12 p14]

permutevar8x32_ps(idx = {0,1,4,5,2,3,6,7})
    把两个 lane 交错开:
    = [p0 p2 p4 p6 p8 p10 p12 p14]     ✓
```

**只要 3 条指令、2 次 load。** 如果按标量 gather 写，光是取数就会吃掉全部收益 ——
这是 stride-2 depthwise 能达到 **1.7×** 加速的关键。

一个容易写错的地方：窗口内 `kx` 的偏移是 `+0/+1/+2`（**相邻列**），
不是 `+0/+2/+4`。`dw_gather2` 负责"每 2 列取 1 个"，相邻列之间的偏移仍由 `p + kx` 决定。

边界推导（对 mbv2 实测形状 `width=208, pad=1, stride=2`）：

```
ow = 104, 循环条件 ox + 8 <= ow  =>  ox <= 96
dw_gather2(p + 2) 读 p[2 .. 17]  =>  最大输入列 = 2*96 + 17 = 209 = mwi - 1 ✓
```

### 4.5 `conv_1x1_avx2`：4 个输出通道分块

这是**全网络最重要的内核**（实测占 mbv2 运行时长的 **78%**）。

```c
for (i = 0; i < N; i += 4) {                  /* 外层按 4 个输出通道分块 */
    /* 内层：c -> y -> x，一次 load 供 4 个输出通道复用 */
    for (c = 0; c < C; c++) {
        float w0 = W[i+0][c], w1 = W[i+1][c], w2 = W[i+2][c], w3 = W[i+3][c];
        for (y) {
            for (x = 0; x + 8 <= ow; x += 8) {
                __m256 v = _mm256_loadu_ps(in + c*plane + y*wi + x);   /* ← 关键：一次 load */
                a0 = fmadd(set1(w0), v, a0);
                a1 = fmadd(set1(w1), v, a1);
                a2 = fmadd(set1(w2), v, a2);
                a3 = fmadd(set1(w3), v, a3);
            }
        }
    }
}
```

**为什么要 4 个输出通道一起算**：1×1 卷积的算术强度很低 —— 每个输入元素只被乘一次。
若一次只算一个输出通道，`loadu` 与 `fmadd` 的比值是 1:1，load 端口先饱和。
4 路共享之后比值变成 1:4，**FMA 的延迟被彻底掩盖**。实测同形状下 **1.5~2.3×** 加速。

**为什么是 4 而不是 8**：8 路需要同时维护 8 个累加器 + 8 个广播权重，
寄存器压力让编译器溢出到栈上，反而变慢。开发中实测过 8 路版本：
在"输出通道多、空间尺寸小"的层（如 `13×13×960 → 320`）会退化到 **0.77×**。

**边界安全性**：

```
loadu(in + c*plane + y*wi + x) 一次读 8 个 float
循环条件 x + 8 <= ow，且本内核要求 wi == ow（输入 pad == 0）
故最大读取下标 = ow - 1，落在行内 ✓
```

尾部按标量补齐。宽度在 MobileNet 系网络里基本都是 8 的倍数（208/104/52/26/13…），
所以尾部循环通常只跑 0~7 次。

### 4.6 `dot4_avx2`：8 路点积

通用路径（非 dw、非 1×1）仍走 im2col + 点积，但点积换成向量版：

```
8 路向量累加 + 双累加器（两路 __m256 交替，进一步拉开依赖）
收尾：0..7 依次塞进 128 位做水平相加，剩余不足 8 个的用标量补进 s1
```

**为什么通用路径能用 8 路而 1×1 只用 4 路**：

| | 通用路径 | 1×1 |
|---|---|---|
| `ftotal` 典型值 | 144 / 576 / 864 / 1728，**都是 8 的倍数** | `C`，任意值 |
| 同时维护的寄存器 | 2 个累加器 + 1 个 load | 4 个累加器 + 4 个广播权重 |
| 结论 | 8 路不浪费且寄存器够 | 4 路才不溢出 |

### 4.7 几何分派的差异（一个真实限制）

avx2 的 depthwise 专用内核**只处理 3×3 且 stride ∈ {1, 2}**：

```c
if (is_dw && filter.size == 3 && (filter.stride == 1 || filter.stride == 2)) {
    ... 专用内核 ...
}
/* 其它 depthwise 落到通用的 im2col + dot4_avx2 路径 */
```

因为专用内核把 9 个权重硬编码成 `w0..w8`（3×3）。其它形状（5×5 depthwise 等）
虽然正确，但会退回到通用路径 —— 性能上不是最优。这是有意的取舍：
写一个"任意 size 的向量 depthwise"需要动态展开，收益不如代码复杂度。

`opt` 的 `conv_depthwise` 则是任意 `size`/`stride` 的标量实现。

---

## 五、三套实现横向对照

| 维度 | orig | opt | avx2 |
|---|---|---|---|
| **BN** | 每元素现场算，近似 rsqrt | 加载时折叠（精确 sqrtf） | 同 opt |
| **im2col** | 每输出像素一次 | 每组一次，被 `n` 摊薄 | 同 opt |
| **写顺序** | 整平面跳跃 | 按行连续 | 按行连续 |
| **累加** | 单链 | 4 链 `dot4` | 8 路向量 + 双累加器 + FMA |
| **depthwise** | 通用 im2col | 专用标量内核（任意 size/stride） | 专用向量内核（仅 3×3，stride 1/2） |
| **1×1** | 通用 im2col | 专用标量内核 | 专用向量内核，4 输出通道分块 |
| **向量宽度** | — | — | 8 float/向量 |
| **CPU 依赖** | 无 | 无 | AVX2 + FMA |
| **可用性** | 任意平台 | 任意平台 | 仅 x86-64 + 运行期支持 |
| **数值角色** | 逐位基线 | 与 orig 差 1e-1（rsqrt）/ 1e-4（精确） | 与 opt 差约 1e-4 |
| **性能角色** | 1×（分母） | 2.3~2.7× | 6.6~8.4× |

加速比来源的大致分解（`opt` 相对 `orig`）：

| 来源 | 贡献 |
|---|---|
| BN 折叠（去掉每元素近似开方） | depthwise / 1×1 层收益最大 |
| 几何特化（跳过 im2col） | 1×1 与 depthwise 层 |
| 一次性 im2col + 按行写 | 通用 3×3 层 |
| `dot4` | 通用路径 |

`avx2` 相对 `opt` 的收益则几乎全部来自三个热内核的向量化，
**1×1 内核是最大头**（占 mbv2 运行时的 78%）。实测数据见 [performance.md](performance.md)。

---

## 六、试过但无效的优化

记录下来，避免后人重走弯路。全部在 AMD Ryzen 7 8745HS（Zen 4）上实测。

### 6.1 1×1 内核的 8 路输出通道分块 —— 反而更慢

| | 结果 |
|---|---|
| 期望 | 更充分地掩盖 FMA 延迟 |
| 实测 | 大输出通道 + 小空间尺寸的层（`13×13×960 → 320`）退化到 **0.77×** |
| 原因 | 8 个累加器 + 8 个广播权重超出 ymm 寄存器数量，编译器溢出到栈；`vpermps`/重载开销吃掉收益 |
| 结论 | **保持 4 路**。4 路在所有实测形状上稳定在 1.5~2.3× |

### 6.2 depthwise 激活函数外提 —— 反而更慢

| | 结果 |
|---|---|
| 期望 | 把 `activate` 从内层循环提到外面，避免逐元素判断 |
| 实测 | `-O3` 下 **0.67~0.75×** |
| 原因 | 外提后要**多遍历一遍输出缓冲**做激活，depthwise 的输出很小但通道很多，多一趟内存往返比内层的无分支 `max` 更贵 |
| 结论 | **保持内层 `VACT`**。`max` 只有 1 条指令，不值得为它加一次 pass |

### 6.3 全局 `-march=native` —— 略慢且破坏可移植性

| 编译选项 | mbv2 416×416 |
|---|---|
| `-mavx2 -mfma`（全局） | **2547 ms** |
| `-march=native` | 2875 ms |

在 Zen 4 上 `-march=native` 反而更慢（推测与自动向量化策略/调度模型有关），
而且会把整个产物绑定到构建机的指令集 —— 在老 CPU 上直接 `SIGILL`，连降级机会都没有。

**最终方案**：文件级 `#pragma GCC target("avx2,fma")` + 运行期 CPU 检测。
构建脚本保持 `-Wall -O3`，不加任何 `-m` 选项。

> 注意：上表是"全局 `-march`"实验的数据，用于说明**为什么不用全局选项**。
> 最终代码不使用任何全局 `-m` 开关，AVX2 指令只出现在 `kernel_conv_avx2.c` 里。

### 6.4 `orig` 每元素现场算 BN 的"优化" —— 属于基线，不改

`orig` 的 `fast_inverse_sqrt` 明显是热点，但如果把它换成 `sqrtf`，
`orig` 就不再逐位复现历史基线 —— 它存在的意义就没了。
需要的场景请用 `-DKERNEL_ORIG_EXACT_RSQRT` 编译开关，而不是改源代码默认值。

---

## 七、如何新增第四套内核

假设要加 `kernels/neon/kernel_conv_neon.c`（ARM NEON）。步骤：

### 步骤 1：在 `kernel_common.h` 注册编号

```c
enum {
    CONV_IMPL_ORIG = 0,
    CONV_IMPL_OPT  = 1,
    CONV_IMPL_AVX2 = 2,
    CONV_IMPL_NEON = 3        /* ← 新增 */
};
void kernel_conv_forward_neon(NET *net, LAYER *ilayer, LAYER *olayer);
```

### 步骤 2：写实现

照抄 `kernels/avx2` 的骨架，把 `__m256` 换成 `float32x4_t`：

```c
#if defined(__GNUC__) && defined(__aarch64__)
#define KERNEL_HAVE_NEON 1
#include <arm_neon.h>
/* AArch64 上 NEON 是基线，不需要 target pragma */
void kernel_conv_forward_neon(NET *net, LAYER *ilayer, LAYER *olayer) { ... }
#else
void kernel_conv_forward_neon(NET *net, LAYER *ilayer, LAYER *olayer) { (void)net; /* 空壳 */ }
#endif
```

### 步骤 3：改派发层 `kernels/common/kernel_common.c`

```c
int kernel_impl_from_name(const char *name) {
    if (!kernel_strcmpi(name, "orig")) return CONV_IMPL_ORIG;
    if (!kernel_strcmpi(name, "opt" )) return CONV_IMPL_OPT;
    if (!kernel_strcmpi(name, "avx2")) return CONV_IMPL_AVX2;
    if (!kernel_strcmpi(name, "neon")) return CONV_IMPL_NEON;   /* ← 新增 */
    return -1;
}
```

`kernel_conv_forward` 加一个 `case`，并在 `kernel_impl_needs_bn_fold` 里
确认新实现走"需要折叠"分支（`return impl != CONV_IMPL_ORIG;` 已自动覆盖）。

### 步骤 4：加编译文件与测试

- `build-win.sh` / `build-linux.sh` 加一行源文件；
- `compare.sh` 的源文件列表也要加（否则链接失败）；
- `test_consistency.c` 的 `impl` 数组加一项，让新实现自动进入逐层对拍。

### 步骤 5：验证清单

```bash
# 1. 与 opt 的数值等价（应落在 1e-4 量级）
./t_cons test.bmp yolo-fastest-1.1.cfg yolo-fastest-1.1.weights

# 2. 确定性（同实现跑两次逐位相同）
#    test_consistency.c 已内置

# 3. 不支持时能降级（在无 NEON 的机器上跑）
```

> **不要**在 `kernel_common.h` 里加 `#include <arm_neon.h>` —— 那会污染其它平台的编译。
> 头文件只在实现文件内部按 `#if` 引入。

---

## 八、写内核的检查清单

新增或修改任何卷积内核时，逐条对照：

| # | 检查项 | 常见错误 |
|---|---|---|
| 1 | 窗口锚点**不加 `+pad`** | 平移一个像素 + 右下越界 |
| 2 | 只写输出内部区域，不碰 pad | 跨帧累积错值 |
| 3 | 向量 load 的最大下标 ≤ `mw - 1` | 行尾越界读（可能不崩，但结果错） |
| 4 | 尾部不足一个向量的部分用标量补齐 | 最后 `ow % 8` 个像素没写 |
| 5 | 1×1 内核要求输入 `pad == 0`（`wi == ow`） | 用 `wi` 当 `ow` 导致错位 |
| 6 | depthwise 判据用 `filter.channels == 1`（已折算成每组通道） | 误用 `groups == channels` |
| 7 | 只用 `data2`/`bias2`，并检查 `bn_fused` | 读到未折叠的副本 → 结果错 |
| 8 | 累加顺序固定、可复现 | 不同优化等级下结果不同 |
| 9 | 不新增 `malloc`（除兜底分支） | `net_forward` 内分配 |
| 10 | 跑 `test_dw` / `test_1x1` / `test_consistency` | — |

---

## 延伸阅读

- [architecture.md](architecture.md) —— 数据布局、pad 策略、锚点约定的完整推导
- [performance.md](performance.md) —— 上述优化的实测收益与 profiling 方法
- [testing.md](testing.md) —— 怎么证明你的内核是对的
