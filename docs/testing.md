# 测试与验证

> 本文说明三个测试程序**各自验证什么**、判定阈值怎么来的、以及怎么用 `rsqrt` 开关
> 把"浮点近似"和"实现错误"分开归因。
>
> 一句话结论：**优化不改变模型输出** —— 三套内核的差异全部落在浮点结合顺序与
> `orig` 自身的近似开方能解释的范围内。

---

## 目录

- [一、三层测试金字塔](#一三层测试金字塔)
- [二、单内核测试：test_dw / test_1x1](#二单内核测试test_dw--test_1x1)
- [三、一致性测试：test_consistency](#三一致性测试test_consistency)
- [四、判定阈值的依据](#四判定阈值的依据)
- [五、rsqrt 隔离法](#五rsqrt-隔离法)
- [六、实测结果](#六实测结果)
- [七、如何判读失败](#七如何判读失败)
- [八、已知的测试覆盖缺口](#八已知的测试覆盖缺口)
- [九、建议的 CI 流程](#九建议的-ci-流程)

---

## 一、三层测试金字塔

```
                    ┌──────────────────────────────┐
                    │  端到端一致性                 │  跑真实模型
                    │  test_consistency.c           │  逐层特征图 + 检测框 + 确定性
                    └──────────────────────────────┘
                  ┌──────────────────────────────────┐
                  │  单内核正确性                     │  一个内核 × 9 种形状
                  │  test_dw.c / test_1x1.c           │  对拍独立写的朴素实现
                  └──────────────────────────────────┘
                ┌──────────────────────────────────────┐
                │  隐式检查                             │  每个内核都靠
                │  pad 边缘不变量 / 锚点约定             │  "只写内部区域 + 不加 +pad"
                └──────────────────────────────────────┘
```

| 层 | 抓什么问题 | 代价 |
|---|---|---|
| 单内核 | 某一种形状下的索引错误、边界处理错误 | < 1 秒 |
| 一致性 | 三套实现互不一致、某层整体错、非确定性 | 数秒（mbv2 上十几秒） |
| 隐式 | 跨帧累积的 pad 污染 | 一致性测试顺带覆盖 |

---

## 二、单内核测试：test_dw / test_1x1

### 2.1 共同思路

两个测试都做同一件事：**用伪随机数据跑被测内核，再用一份独立写的朴素实现算一遍，逐元素比对。**

```c
#include "kernels/opt/kernel_conv_opt.c"   /* 直接包含 .c 以访问 static 内核函数 */
```

因为是 `#include` 源文件而不是链接目标文件，`static` 的 `conv_depthwise` / `conv_1x1`
可以直接调用，不需要为了测试把它们改成非 `static`。

### 2.2 `test_dw.c`：depthwise 卷积

覆盖 9 种形状，**每一维都挑了会让 bug 暴露的取值**：

| 用例 | W×H | C | size | stride | 输出 pad | 覆盖要点 |
|---|---|---|---|---|---|---|
| 1 | 16×16 | 8 | 3 | 1 | 0 | 基线 |
| 2 | 16×16 | 8 | 3 | 1 | 1 | 输出带 pad（验证"输出 pad 属于下一层"） |
| 3 | 16×16 | 8 | 3 | 2 | 0 | stride 2 基线 |
| 4 | 16×16 | 8 | 3 | 2 | 1 | stride 2 + 输出 pad |
| 5 | 160×160 | 8 | 3 | 1 | 1 | 大平面（接近真实首层） |
| 6 | 160×160 | 8 | 3 | 2 | 1 | 大平面 + 下采样 |
| 7 | 10×10 | **224** | 3 | 1 | 0 | 大通道数（接近 MobileNet 深层） |
| 8 | **7×9** | 5 | 3 | 2 | **2** | 非方阵 + 奇数尺寸 + 非对称 pad |
| 9 | **20×12** | 32 | 3 | 2 | 1 | 非方阵 + 中等通道 |

输入缓冲**故意填满随机值再显式清零边缘区**，模拟引擎的 margin 约定：

```c
for (i = 0; i < mwi*mhi*C; i++) in[i] = rnd();          /* 先全随机 */
for (每个边缘位置) in[...] = 0.0f;                       /* 再清 pad */
```

这样如果内核错误地读了 pad 区，会读到 0（而不是随机值），差异就更难藏。

**关键教训（写在文件头注释里）**：

> 参考实现对窗口锚点使用 **padded 数组坐标** —— 输出像素 `(x,y)` 的窗口是
> `padded[y*STRIDE .. y*STRIDE+SIZE-1]` 行，**不加 `+PAD` 偏移**。
> 曾因参考实现与内核犯**同样的** `+PAD` 错误而漏测。

这是"对拍测试"的经典陷阱：**参考实现不能和被测定实现共享同一个错误假设**。
写参考实现时应该从"卷积的数学定义"出发，而不是从"被测定实现的代码"出发。

### 2.3 `test_1x1.c`：pointwise 卷积

单个形状 `16×16×8 → 4`，但特意让**输出槽位带 pad**（`OPAD = 1`）：

```c
MATRIX mi = { W, H, C, 0,    in  };   /* 1x1 内核要求输入 pad == 0 */
MATRIX mo = { W, H, N, OPAD, out };   /* 输出带 pad，验证写入范围 */
```

验证两件事：

1. 数值正确：`maxdiff` 应为 `0.000000`（`N*C = 32` 次乘加，单精度下顺序一致时无误差）；
2. **内核对输出的写入范围**：参考值从 `out[OPAD + y][OPAD + x]` 取，
   即内核必须把结果写在内部区域，而不是平面偏移处。

### 2.4 编译与运行

```bash
gcc -O2 -I. test_dw.c  kernels/common/kernel_shared.c -lm -o t_dw  && ./t_dw
gcc -O2 -I. test_1x1.c kernels/common/kernel_shared.c -lm -o t_1x1 && ./t_1x1
```

> 需要 `kernel_shared.c`，因为内核会调用 `activate()` 与 `kernel_im2row()`。
> `-O2` 就够 —— 测试目标是**正确性**，不是性能；用 `-O2` 还能让编译器不做
> 过度重排，便于定位。

期望输出：

```
DW TEST ALL PASS
conv_1x1 maxdiff=0.000000
```

---

## 三、一致性测试：test_consistency

`test_consistency.c`（约 400 行）是这个项目里最重要的验证工具。

### 3.1 它要回答的问题

`opt` / `avx2` 相对 `orig` 改了三件事，**每一件都可能改变浮点结果**：

| # | 改动 | 影响 |
|---|---|---|
| 1 | **BN 折叠方式** | `orig` 每元素现场算并用 `fast_inverse_sqrt` 近似（≈0.2% 相对误差）；`opt` 加载时折叠并用精确 `sqrtf` |
| 2 | **点积累加顺序** | `orig` 单链；`opt` 4 链；`avx2` 8 路向量 + 双累加器 + FMA（一次舍入而非两次） |
| 3 | **1×1 / depthwise 走专用内核** | 累加顺序与数学定义一致，但仍有上述舍入差异 |

所以必须把"**内核是否正确**"与"**浮点顺序不同**"分开度量。

### 3.2 快照机制

```c
typedef struct {
    int    layer_num;
    int    w[MAX_SNAP_LAYERS], h[...], c[...], pad[...];
    float *data[MAX_SNAP_LAYERS];   /* 每层特征图副本 */
    int    bbox_num;
    BBOX  *bbox;                    /* 检测框副本 */
} SNAPSHOT;
```

`MAX_SNAP_LAYERS = 512`，覆盖 mbv2 的 131 个输出槽位绰绰有余。

两个容易踩的坑（代码里专门注明了）：

| 坑 | 说明 |
|---|---|
| **不要比 slot0** | `layer_list[0].matrix` 是输入缓冲，且被 `bbox_list` 复用；前向结束后里面存的是**检测框**而不是输入图像。所以从 `slot1` 开始比 |
| **检测框必须立刻复制** | 下一次 `net_input` 会把这块内存 `memset` 掉 |

### 3.3 比较策略

```
 ┌──────────────────────────────────────────────────────────────┐
 │ 1. 三个实现各跑一次，各留一份快照                              │
 │    orig / opt / avx2 —— 通过 net_set_conv_impl 切换           │
 ├──────────────────────────────────────────────────────────────┤
 │ 2. [严格] opt vs avx2                                         │
 │    两者都折叠了 BN，只有向量宽度与链数不同 => 必须高度等价      │
 │    CMP_HARD => 直接判失败（avx2 有实现错误）                   │
 ├──────────────────────────────────────────────────────────────┤
 │ 3. orig vs opt / avx2                                         │
 │    含 rsqrt 近似项，CMP_HARD 是否算失败取决于构建模式           │
 ├──────────────────────────────────────────────────────────────┤
 │ 4. 确定性自检（每个实现）                                      │
 │    连跑两次必须逐位相同                                       │
 └──────────────────────────────────────────────────────────────┘
```

**第 2 步是最严格、最能暴露 bug 的一组**。理由：`opt` 和 `avx2` 的 BN 折叠方式完全相同，
差异只剩向量宽度与累加链数 —— 任何超出 1e-3 的差异都只能是实现错误。

### 3.4 三级返回码

```c
enum {
    CMP_EQUIV = 0,   /* 数值等价                       */
    CMP_SOFT  = 1,   /* 略超等价阈值但未达报警线       */
    CMP_HARD  = 2    /* 达到报警线，需要归因           */
};
```

不直接返回"通过/失败"的原因：`orig` 与 `opt` 的对比里混着"BN 折叠方式不同"这一**已知项**，
算不算失败取决于当前 `orig` 用的是哪种 rsqrt。由调用方结合编译开关判定才准确。

比较逻辑：

```c
if (worst_abs <= TENSOR_TOL_EQUIV)      return CMP_EQUIV;   /* 1e-3 */
if (n_over == 0)                        return CMP_SOFT;    /* 无层超过 1e-2 */
return CMP_HARD;
```

注意 `n_over`：**多少层超过报警线**。一层超线可能是数值巧合，几十层同时超线必然是系统性问题。

### 3.5 相对误差的统计陷阱

```c
if (aa > 1e-3f) {           /* 只统计 |a| > 1e-3 的元素 */
    float r = d / aa;
    if (r > mrel) mrel = r;
}
```

接近 0 的分母会把**毫无意义的微小绝对误差**放大成巨大的"相对误差"。
例如 `a = 1e-9, b = 2e-9`，绝对差只有 `1e-9`，相对差却是 100%。
这是统计陷阱，不能用来判定。

### 3.6 检测框比较

```c
if (ref->bbox_num != tst->bbox_num) {
    printf("  -> 数量不同（后处理阈值敏感性，请结合特征图误差判断）\n");
    return;
}
```

**数量不同不计为实现错误**：后处理含 `sigmoid` / `exp` / 阈值比较 / 排序，
输入上的微小差异完全可能让某个候选框在阈值附近翻转。
归因要看**特征图误差**，而不是框数。

坐标差用**相对量级**衡量：

```
最大坐标/分数差 = xxx，坐标量级 = xxx，相对差 = mx/scale
```

因为本仓库的 mbv2 模型有 [已知的 YOLO 解码缺陷](known-issues.md)，坐标可达 `1e5` 量级 ——
此时 `357` 的绝对差只有 0.3%，属同一现象，不能按"坐标必须一致"来判。

### 3.7 确定性自检

```c
net_input(...); net_forward(net); snapshot_take(net, a);
net_input(...); net_forward(net); snapshot_take(net, b);
/* a 与 b 必须逐位相同 */
```

专门抓两类隐蔽缺陷：

| 缺陷 | 特征 |
|---|---|
| 读了未初始化的缓冲 | 首次运行读到垃圾，第二次读到上次的残留 → 两次不同 |
| 依赖上次运行的残留状态 | 同上 |

它们**不会在数值上露头**（两次结果都"看起来合理"），但会让结果不可复现 ——
一旦不可复现，所有基准数据都失去意义。

---

## 四、判定阈值的依据

```c
#define TENSOR_TOL_EQUIV  1e-3f   /* 数值等价判据 */
#define TENSOR_TOL_ERROR  1e-2f   /* 实现错误报警线 */
```

特征图幅值通常 `O(1) ~ O(10)`，通道数几十到上百。单精度下各种误差源的量级：

| 误差源 | 典型最大绝对误差 | 是否正常 |
|---|---|---|
| 累加顺序不同（`dot4` / 8 路向量 vs 单链） | `1e-6 ~ 1e-4` | ✅ 正常 |
| BN 折叠方式不同**但都用精确 sqrt** | 同量级 `1e-6 ~ 1e-4` | ✅ 正常 |
| FMA 一次舍入 vs 两次舍入 | `1e-6 ~ 1e-4` | ✅ 正常 |
| 近似 rsqrt（0.2% 相对误差）经多层放大 | `1e-1 ~ 1e0` | ⚠️ 已知近似 |
| 索引/布局/激活漏项等**实现错误** | `≥1e-1`，**且伴随大量层同时超标** | ❌ 必须修 |

由此定：

- **`1e-3`** 作为"数值等价"判据。相对误差约 `1e-5`，比浮点顺序差异的上界高一个数量级
  （不会误报），又比近似 rsqrt 的 `1e-1` 低两个数量级（能区分）。
- **`1e-2`** 作为"实现错误"报警线。到了这个量级，再说是"顺序差异"就不合理了。

> 阈值不是"约等于 0"的严格检查。要求逐位相等会把所有优化都判为失败 ——
> 因为只要改变累加顺序，浮点结果就必然不同。

---

## 五、rsqrt 隔离法

这是整个验证体系的关键技巧。

### 5.1 问题

`orig` 的 `fast_inverse_sqrt`（Quake III 魔数 + 一次牛顿迭代）有约 **0.2%** 相对误差。
逐层放大后变成 `1e-1` 量级的绝对偏差。这**不是**内核实现错误，
而是历史基线自身的近似。

但只看到 `5.78e-01` 这个数字，你无法确定它是近似还是 bug。

### 5.2 方法：两种构建各跑一次

```c
/* kernels/orig/kernel_conv_orig.c */
#ifdef KERNEL_ORIG_EXACT_RSQRT
  #define ORIG_INV_SQRT(x) (1.0f / sqrtf(x))              /* 精确 */
#else
  #define ORIG_INV_SQRT(x) kernel_fast_inverse_sqrt(x)    /* 魔数近似（默认） */
#endif
```

| 构建 | 编译命令 | `orig` 的 BN 归一化 |
|---|---|---|
| **A**（默认） | `gcc -O3 ... -o tc_a` | 魔数近似，与历史基线逐位一致 |
| **B** | `gcc -O3 -DKERNEL_ORIG_EXACT_RSQRT ... -o tc_b` | 精确 `sqrtf` |

### 5.3 判据

```
构建 B 下 orig vs opt 的误差降到 1e-4 量级  =>  原差异完全由近似解释（优化正确）
构建 B 下 orig vs opt 仍然很大            =>  存在真正的实现错误
```

**逻辑**：构建 B 里三套实现的 BN 数学形式完全相同，差异只剩累加顺序 ——
而累加顺序的差异不可能超过 `1e-4`。如果还大，就一定是索引/布局/激活的问题。

### 5.4 为什么不干脆把默认值改成精确

因为 `orig` 的职责是"**逐位复现历史基线**"。基线正是用近似值算出来的 ——
改了它，历史对比就失去参照。需要精确值的场景用编译开关，不改源代码默认行为。

---

## 六、实测结果

环境：AMD Ryzen 7 8745HS / GCC 16.2 / `-Wall -O3`。

### 6.1 逐层特征图最大绝对误差

| 对比 | yolo-fastest-1.1 | mbv2_yolov3_face | 判定 |
|---|---|---|---|
| `opt` vs `avx2` | **8.333e-05** | **1.049e-04** | 数值等价 |
| `orig` vs `opt`（构建 A，近似 rsqrt） | 5.776e-01 | 1.209e+00 | 由近似解释 |
| `orig` vs `opt`（构建 B，精确 rsqrt） | **1.044e-04** | **2.289e-04** | 数值等价 |

### 6.2 检测框数量

| 构建 | yolo-fastest | mbv2-face |
|---|---|---|
| A（近似） | 3 vs 3 | 962 vs 971 |
| B（精确） | 3 vs 3 | **971 vs 971** |

**构建 B 下 mbv2 的框数完全一致（971 vs 971）** —— 这比特征图误差更有说服力：
连后处理的非线性放大都没能让结果翻转。

构建 A 的 962 vs 971 差异，就是 `1e-1` 量级的特征图误差经 `sigmoid` / 阈值后
让 9 个边缘候选框翻转的结果。

### 6.3 确定性

三套实现在两种构建下，各自连跑两次均**逐位相同**。

### 6.4 结论

> **优化不改变模型输出。** 全部差异可由浮点结合顺序与 `orig` 的近似开方解释。

---

## 七、如何判读失败

### 7.1 `opt` vs `avx2` 出现 `CMP_HARD`

**这是真 bug，必须修。** 两者唯一的区别是向量宽度与累加链数，不可能产生 `1e-2` 级的差异。

排查顺序：

1. 看**哪一层**误差最大（输出里的 `最差层=N`）→ 对照 `net_dump` 找出该层的类型与尺寸；
2. 如果是最差层是 depthwise stride 2 → 查 `dw_gather2` 的边界推导
   （见 [kernels.md 第 4.4 节](kernels.md)）；
3. 如果是 1×1 → 查 `wi == ow` 的前提（输入 pad 必须为 0）；
4. 如果是最深的小特征图层 → 查尾部标量循环是否写全 `ow % 8` 个像素；
5. 通用层 → 查 `dot4_avx2` 的 8 路收尾与水平相加。

### 7.2 `orig` vs `opt` 出现 `CMP_HARD`（构建 A）

**先别急着改代码**，按顺序做：

1. 用 `-DKERNEL_ORIG_EXACT_RSQRT` 重编 → 重跑；
2. 如果误差降到 `1e-4` 量级 → **已知行为，不是 bug**；
3. 如果仍然很大 → 真错误，按 7.1 的流程排查，并且此时 `opt` vs `avx2`
   应该也会报警（因为两者共用同一套几何分派逻辑）。

### 7.3 确定性自检失败

```
!! 非确定性: layer 37 maxabs=3.2e-08
```

说明有代码读了**未初始化**或**上次残留**的内存。常见来源：

| 来源 | 检查点 |
|---|---|
| `cnntempbuf` 只填了部分区域 | im2col 是否覆盖了完整的 `ftotal × ow × oh` |
| 尾部像素没写 | 标量补齐循环的边界 |
| pad 区被误改 | 内核的写入范围 |
| `foldbuf` 的片区没有全部赋值 | `net_bn_fold` 是否漏了某层 |

> `maxabs` 很小（如 `1e-8`）也要报错 —— 因为确定性要求是**逐位相同**，
> 任何非零差异都意味着读了不该读的东西。

---

## 八、已知的测试覆盖缺口

诚实记录，方便后续补测。

| # | 缺口 | 现状 | 补法 |
|---|---|---|---|
| 1 | **AVX2 内核没有单内核对拍测试** | `test_dw.c` / `test_1x1.c` 通过 `#include kernel_conv_opt.c` 只测 **opt 的标量内核**；AVX2 只被 `test_consistency` 的 opt-vs-avx2 间接覆盖 | 写 `test_avx2.c`，同样 `#include` avx2 的实现文件，对拍朴素实现 |
| 2 | 单内核测试只覆盖 3×3 | `test_dw.c` / `test_1x1.c` 的 `size` 都是 3 | 加 `size = 1/5/7` 的用例（`opt` 的 `conv_depthwise` 支持任意 size） |
| 3 | `compare.sh` 依赖 `/tmp`、`md5sum` 等 Unix 工具 | 只做过 `sh -n` 语法检查，**未在纯 Windows 环境实际执行** | 用 Git Bash 或 WSL 跑一次 |
| 4 | 没有性能回归测试 | 只有人工对比 | 把 [performance.md](performance.md) 的基准脚本化，设一个宽松阈值（如 ±20%） |
| 5 | mbv2 的 YOLO 解码缺陷只验证了"现象"，没有修复后的基线 | 见 [known-issues.md](known-issues.md) | 修 `layer_yolo.c` 后重录基线 |

---

## 九、建议的 CI 流程

```bash
#!/bin/sh
# 建议的最小 CI 脚本
set -e

# 1. 编译（应无新增警告；当前有一处历史遗留警告，见下方说明）
gcc -Wall -O3 bmpfile.c parser.c layer_conv.c layer_pool.c layer_misc.c \
    layer_yolo.c nms.c net.c main.c \
    kernels/common/kernel_shared.c kernels/common/kernel_common.c \
    kernels/orig/kernel_conv_orig.c kernels/opt/kernel_conv_opt.c \
    kernels/avx2/kernel_conv_avx2.c -o test -lm

# 2. 单内核
gcc -O2 -I. test_dw.c  kernels/common/kernel_shared.c -lm -o t_dw  && ./t_dw
gcc -O2 -I. test_1x1.c kernels/common/kernel_shared.c -lm -o t_1x1 && ./t_1x1

# 3. 一致性（构建 A：默认）
gcc -O3 test_consistency.c bmpfile.c parser.c layer_conv.c layer_pool.c \
    layer_misc.c layer_yolo.c nms.c net.c \
    kernels/common/kernel_shared.c kernels/common/kernel_common.c \
    kernels/orig/kernel_conv_orig.c kernels/opt/kernel_conv_opt.c \
    kernels/avx2/kernel_conv_avx2.c -o t_cons -lm
./t_cons test.bmp yolo-fastest-1.1.cfg yolo-fastest-1.1.weights

# 4. 一致性（构建 B：orig 也精确）—— 这一关必须通过
gcc -O3 -DKERNEL_ORIG_EXACT_RSQRT test_consistency.c bmpfile.c parser.c \
    layer_conv.c layer_pool.c layer_misc.c layer_yolo.c nms.c net.c \
    kernels/common/kernel_shared.c kernels/common/kernel_common.c \
    kernels/orig/kernel_conv_orig.c kernels/opt/kernel_conv_opt.c \
    kernels/avx2/kernel_conv_avx2.c -o t_cons_b -lm
./t_cons_b test.bmp yolo-fastest-1.1.cfg yolo-fastest-1.1.weights
```

**关键**：第 4 步（构建 B）是唯一一个**必须零容忍**的关卡 ——
它的返回码非 0 就意味着三套实现真的不一致。

> **关于第 1 步的警告**：当前 `-Wall -O3` 下有**一处历史遗留警告**：
>
> ```
> layer_pool.c:38:25: warning: variable 'oy' set but not used [-Wunused-but-set-variable]
> ```
>
> `layer_avgmaxpool_forward` 里的 `oy` 只在 `for (iy=0, oy=0; ...; oy++)` 中被赋值，
> 从未被读取（行偏移用的是 `datao` 指针推进）。删掉它即可零警告，属纯清理。
> CI 里建议只拦截**新增**警告，不要因为它把构建判红。

---

## 延伸阅读

- [kernels.md](kernels.md) —— 被测试的内核的具体算法与边界推导
- [architecture.md](architecture.md) —— 数据布局与不变量（测试正是在验证这些）
- [known-issues.md](known-issues.md) —— 已知未修的问题与测试缺口的详细说明
