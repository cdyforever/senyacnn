# 已知问题与限制

> 这份文档记录**已知但未修**的问题、**能力边界**，以及一些**容易被误判为 bug 的正常行为**。
>
> 目的很实际：避免下一个人花两天时间重新发现"这个模型本来就跑不出正确结果"。

---

## 目录

- [总览](#总览)
- [一、mbv2 YOLO 解码缺陷（🔴 高）](#一mbv2-yolo-解码缺陷-高)
- [二、cfg 里的未知段被静默丢弃（🔴 高）](#二cfg-里的未知段被静默丢弃-高)
- [三、权重文件缺失或过短时静默使用未初始化内存（🟡 中）](#三权重文件缺失或过短时静默使用未初始化内存-中)
- [四、不支持的激活函数被当成 linear（🟡 中）](#四不支持的激活函数被当成-linear-中)
- [五、能力边界（🟢 设计取舍）](#五能力边界-设计取舍)
- [六、容易误解的正常行为（⚪ 不是 bug）](#六容易误解的正常行为-不是-bug)
- [七、修复优先级建议](#七修复优先级建议)

---

## 总览

| # | 问题 | 严重度 | 本次内核重构是否引入 | 影响面 |
|---|---|---|---|---|
| 1 | mbv2 的 YOLO 解码与本 cfg 的数据布局不匹配 | 🔴 高 | ❌ 否（`orig` 同样） | 该模型输出垃圾框 |
| 2 | cfg 里的未知段被静默丢弃，导致后续层下标错位 | 🔴 高 | ❌ 否 | 换模型即可能触发 |
| 3 | 权重文件缺失/过短时不报错，用未初始化内存推理 | 🟡 中 | ❌ 否 | 静默给出无意义结果 |
| 4 | 不支持的激活（mish/swish）按 linear 处理 | 🟡 中 | ❌ 否 | 换模型即可能触发 |
| 5 | depthwise AVX2 仅支持 3×3；route 仅 4 路；anchor 仅 9 个 …… | 🟢 低 | ✅ 部分是 | 性能而非正确性 |
| 6 | `orig` 与 `opt` 差 1e-1；mbv2 框数 962 vs 971 | ⚪ 不是 bug | ✅ 是 | 已归因，见 [testing.md](testing.md) |

> **重要**：第 1 项在本仓库的 `orig` 路径上**同样存在**。它不是这次内核重构引入的回归，
> 而是 `layer_yolo.c` 与这个特定 `mbv2_yolov3_face.cfg` 之间的历史不匹配。

---

## 一、mbv2 YOLO 解码缺陷（🔴 高）

### 1.1 现象

加载 `mbv2_yolov3_face.cfg` 后，检测结果毫无意义：

- 输出 **962 ~ 971** 个框（一张只有一张人脸的图）；
- 坐标可达 **±1e5** 量级（远超图像尺寸 416）；
- 分数分布混乱，NMS 后仍然一大堆。

而 `yolo-fastest-1.1` 正常输出 3 个框（测试图里确实有 3 个目标）。

### 1.2 证据：cfg 的三个 yolo 头

| yolo 层（cfg 行） | 前驱 conv 的 `filters` | `mask` | 锚框数 | 每锚框通道 = filters / 锚框数 | 隐含类数 = 每锚框通道 − 5 | cfg 写的 `classes` |
|---|---|---|---|---|---|---|
| ~638 | **48** | `6,7,8` | **3** | `48/3 = 16` | `16−5 = 11` | **1** |
| ~747 | **32** | `4,5` | **2** | `32/2 = 16` | `16−5 = 11` | **1** |
| ~861 | **64** | `0,1,2,3` | **4** | `64/4 = 16` | `16−5 = 11` | **1** |

三行都指向同一个结论：**这份权重是按"每锚框 16 通道 = 11 类"训练出来的**，
但 cfg 里把 `classes` 写成了 `1`。

同时注意 `mask` 的锚框数分别是 **3 / 2 / 4**，**不是**统一的 3。

### 1.3 代码：`layer_yolo.c` 的两处不匹配

```c
for (k = 0; k < 3; k++) {                                        /* L34 */
    int dstart = k * (4 + 1 + ilayer->class_num), cindex = 0;     /* L35 */
    float bs = get_matrix_data(&ilayer->matrix, j, i, dstart + 4);   /* objectness */
    float cs = get_matrix_data(&ilayer->matrix, j, i, dstart + 5);   /* 类别 0 */
    for (l = 1; l < ilayer->class_num; l++) {                     /* 只扫 1 个"类" */
        ...
    }
```

| 缺陷 | 位置 | 应该是什么 |
|---|---|---|
| **锚框数硬编码为 3** | L34 `k < 3` | `k < ilayer->anchor_num`（本模型是 3 / 2 / 4） |
| **每锚框步长按 `classes` 算** | L35 `4 + 1 + class_num`，得 **6** | 实际是 **16** |

具体后果（以第 1 个头为例）：

```
真实张量布局（filters = 48 通道）：
  ch 0..15  : 锚框 0 的 [tx ty tw th obj c0..c10]
  ch 16..31 : 锚框 1
  ch 32..47 : 锚框 2

代码实际读的通道（dstart = k*6）：
  k=0 -> dstart=0 : tx ty tw th obj c0            ✅ 只有 objectness 对了
  k=1 -> dstart=6 : 落进锚框 0 的类别分数区        ❌ 把类别分数当 tx ty tw th
  k=2 -> dstart=12: 落进锚框 0 的最后一小段        ❌
```

被当成 `tw` / `th` 的值原本是类别分数 `sigmoid ∈ (0,1)` 附近的原始 logit，
`exp(tw)` 一旦碰上稍大的正数就会放大成 `1e4` 量级 —— 这就是坐标 ±1e5 的来源。

### 1.4 为什么 yolo-fastest 没问题

| 项 | yolo-fastest-1.1 | 与代码假设是否一致 |
|---|---|---|
| `mask` 锚框数 | `3,4,5` / `0,1,2` → 都是 **3** | ✅ 与 `k < 3` 一致 |
| `classes` | **80** | ✅ |
| `filters` | **255** = `3 × (4+1+80)` | ✅ 与 `4+1+classes` 一致 |

**完全自洽**，所以两个头的解码都正确。

### 1.5 怎么修（两个可选方向）

**方向 A：按张量实际布局解码（推荐，改代码）**

```c
/* 每锚框通道数由 filters / 锚框数 推出，不依赖 cfg 的 classes 字段 */
int dstep    = ilayer->filter.n / ilayer->anchor_num;   /* 例如 48/3 = 16 */
int nclasses = dstep - 5;                               /* 例如 11 */

for (k = 0; k < ilayer->anchor_num; k++) {
    int dstart = k * dstep;
    ...
    for (l = 1; l < nclasses; l++) { ... }
}
```

优点：无论 cfg 的 `classes` 写对没写对，都能按**权重实际形状**正确解码。
缺点：`filter.n` 在 `[yolo]` 层上是否可用需要确认 —— `[yolo]` 不解析 `filters`，
得从前驱 conv 层取（或让解析器把 `filters` 也存进 yolo 层）。

**方向 B：改 cfg 的 `classes`（改数据）**

把三处 `classes=1` 改成 `classes=11`。

优点：改动最小。缺点：`classes` 语义上应该是"这个模型检测几类"，
一个**人脸模型检测 11 类**说明权重文件本身可能来自另一个用途的项目 ——
改 cfg 只是让它"自洽"，不保证语义正确。

**方向 C：两者都做**

先按 A 让代码健壮，再按 B 校数据。**这是最终应该做的**。

### 1.6 为什么这次没有顺手修

三个理由，都是有意为之：

1. **会改变输出**：修完 mbv2 的框数与坐标都会变，历史基线全部失效，
   必须重新录制一套参照值 —— 这属于独立的变更，不该混在"内核重构"里；
2. **正确性归属不明**：cfg 的 `classes` 与权重形状本身就矛盾（见 1.5 方向 B），
   "修好之后应该输出什么"没有客观答案；
3. **不影响本次交付的核心主张**：本次要证明的是"优化不改变模型输出"。
   这个缺陷在 `orig` 上同样存在，因此它**不干扰**三套内核的一致性对比 ——
   `test_consistency` 在构建 B 下给出 `971 vs 971` 完全一致，足以证明这一点。

---

## 二、cfg 里的未知段被静默丢弃（🔴 高）

### 2.1 问题

`parser.c` 的段类型判定是一串 `strstr(pstart, "[xxx]") == pstart` 的 `else if` 链，
**最后没有 `else` 分支**：

```c
if      (pstart 匹配 [conv]...)       { ... }
else if (pstart 匹配 [avg]/[max]...)  { ... }
else if (pstart 匹配 [upsample]...)   { ... }
else if (pstart 匹配 [dropout]/[shortcut]...) { ... }
else if (pstart 匹配 [route]...)      { ... }
else if (pstart 匹配 [yolo]...)       { ... }
/* ← 没有 else，也没有任何提示 */
pstart = pend;
```

### 2.2 后果比"少一层"严重得多

1. `layercur` **不递增**，该段不占槽位；
2. 但后续所有 `[route]` / `[shortcut]` 的 `layers=` / `from=` 都是**相对下标**，
   一旦中间少了一层，**所有后续下标都错位**；
3. 全程**零警告**，程序正常跑完，只是结果不对。

### 2.3 哪些段会触发

| 段名 | 现状 |
|---|---|
| `[reorg]` / `[reorg3d]` | 未实现 → 静默丢弃 |
| `[region]`（YOLOv2 头） | 未实现 → 静默丢弃 |
| `[Gaussian_yolo]` | 未实现 → 静默丢弃 |
| `[scale_channels]` / `[sam]` | 未实现 → 静默丢弃 |
| `[connected]` / `[softmax]` | 未实现 → 静默丢弃 |
| `[batchnorm]`（独立 BN 层） | 未实现 → 静默丢弃 |
| `[cost]` / `[detection]` | 未实现 → 静默丢弃 |

### 2.4 修法

在 `else` 分支里**直接报错退出**，而不是继续：

```c
} else {
    printf("unsupported cfg section: %.*s\n", (int)(pend - pstart), pstart);
    /* 早退：静默错位比直接失败更难查 */
    ...
}
```

代价是某些"用来训练才需要的段"（如 `[cost]`）也会让加载失败，
所以可以先维护一张**可忽略段白名单**，白名单外的一律报错。

---

## 三、权重文件缺失或过短时静默使用未初始化内存（🟡 中）

### 3.1 问题

```c
/* parser.c */
net->weight_buf = malloc(net->weight_size * sizeof(float));   /* ← malloc，不是 calloc */

if ((fp = fopen(fweights, "rb"))) {
    fseek(fp, sizeof(WEIGHTS_FILE_HEADER), SEEK_SET);
    fread(net->weight_buf, 1, net->weight_size * sizeof(float), fp);
    fclose(fp);
}
/* ← fopen 失败时没有任何提示，weight_buf 保持未初始化 */
```

### 3.2 三种触发方式

| 情形 | 结果 |
|---|---|
| 权重文件不存在 | `fopen` 返回 `NULL`，`weight_buf` **未初始化**，照常推理 |
| 文件比期望短 | `fread` 只填了一部分，剩余部分未初始化 |
| 文件是另一个模型的 | 能读满，但语义完全错 —— 这个其实无法检测 |

### 3.3 为什么危险

- `malloc` 拿到的内存**不是零**，里面是上一个使用者的残留数据；
- 首次运行与后续运行的"垃圾值"可能不同 → 表现为**非确定性**；
- 输出完全无意义，但程序不崩溃、不报错，看起来像"模型效果差"。

`test_consistency.c` 的确定性自检能抓到一部分（如果垃圾区被读到两次会不同），
但如果 OS 恰好给了同一块零页，就抓不到。

### 3.4 修法

```c
fp = fopen(fweights, "rb");
if (!fp) { printf("cannot open weights: %s\n", fweights); net_free(net); return NULL; }
fseek(fp, sizeof(WEIGHTS_FILE_HEADER), SEEK_SET);
if (fread(net->weight_buf, 1, nbytes, fp) != nbytes) {
    printf("short weights file: %s (need %d bytes)\n", fweights, nbytes);
    fclose(fp); net_free(net); return NULL;
}
fclose(fp);
```

顺带一提：`calloc` 也能消除"未初始化"这一半问题，但会掩盖"文件缺失"这个错误 ——
**显式报错更好**。

---

## 四、不支持的激活函数被当成 linear（🟡 中）

```c
/* parser.c */
static int get_activation_type_int(char *str)
{
    static const char *STR_TAB[] = { "linear", "relu", "leaky", NULL };
    for (...) if (strstr(str, STR_TAB[i]) == str) return i;
    return -1;              /* ← 未知返回 -1 */
}

/* kernels/kernel_common.h */
static __inline float kernel_actv(float x, int type)
{
    switch (type) {
    case ACTIVATE_TYPE_RELU : ...
    case ACTIVATE_TYPE_LEAKY: ...
    default                 : return activate(x, type);   /* -1 落到这里 */
    }
}

/* kernels/common/kernel_shared.c */
float activate(float x, int type) { ... default: return x; }   /* -1 → 恒等 */
```

所以 cfg 里写 `activation=mish` / `swish` / `elu` / `logistic`，会被**静默当成 linear**。

### 4.1 影响判断

现代 YOLO 系列的 cfg 常用 `mish` / `swish`。用这类模型会得到"数值合理但结果不对"的输出 ——
比崩溃更难发现。

### 4.2 修法

两个层次：

| 层次 | 做法 |
|---|---|
| 最低要求 | `get_activation_type_int` 返回 -1 时**打印警告并指出层号**，让用户知道 |
| 完整体 | 实现 `mish` / `swish`（标量版 3 行，AVX2 版需要 `exp` 的向量近似，成本较高） |

注意 `sigmoid` 在 cfg 里出现时**不是**给卷积层用的 —— 它只由 `[yolo]` 解码内部调用
`activate(bs, ACTIVATE_TYPE_SIGMOID)`，与 cfg 的 `activation=` 字段无关。

---

## 五、能力边界（🟢 设计取舍）

这些是**有意的限制**，不是缺陷。列出来是为了让使用者提前判断能不能用。

### 5.1 算子层

| 项 | 限制 | 说明 |
|---|---|---|
| 支持的层类型 | `conv` / `avgpool` / `maxpool` / `upsample` / `dropout` / `shortcut` / `route` / `yolo` | 见 [README](../README.md#支持的层类型) |
| 支持的激活 | `linear` / `relu` / `leaky`（+ 内部 `sigmoid`） | 见第四节 |
| **`route` 最多 4 路** | `depend_list[4]` | 解析循环是 `for (i=0; i<4 && ...)`，第 5 路起被丢弃 |
| `route` 要求各路空间尺寸一致 | 输出宽高取**最后一路** | 与 darknet 的 `route` 语义一致（不做 resize） |
| **`anchors` 最多 9 对** | `anchor_list[9][2]` | YOLOv3 标准是 9 对，够用 |
| `maxpool` / `avgpool` 无线条边界 pad | 输入 pad 由上游 conv 决定 | `matrix_fill_pad` 按**槽位自己的类型**填值 |
| `upsample` 仅最近邻 | 无 `bilinear` | darknet 的 `upsample` 默认也是最近邻 |
| `dropout` 输出槽位别名输入 | 不占 arena 空间 | 推理期恒等映射 |

### 5.2 数值与平台

| 项 | 限制 |
|---|---|
| 精度 | 仅单精度 `float`，无 `fp16` / `int8` / 量化 |
| 线程 | **单线程**，无 OpenMP / pthread |
| 依赖 | 仅 libc + libm，无 BLAS / SIMD 库 |
| 输入 | 单张 BMP（24bpp），单输入单输出 |
| 设备 | CPU only |
| AVX2 | 仅 x86-64 + GCC/Clang；其它平台自动降级到 `opt` |
| `depthwise` AVX2 专用内核 | **仅 3×3 且 stride ∈ {1,2}**；其它形状退回通用 im2col 路径（正确但不最优） |
| `1x1` 专用内核 | 要求 `stride == 1 && pad == 0 && groups == 1` |

### 5.3 后处理

| 项 | 现状 |
|---|---|
| NMS 类型 | 硬阈值 IoU（`0.5`），无 soft-NMS / DIoU-NMS |
| NMS 分母模式 | `min = 1`：取交集 / **较小框面积**（人脸场景常用），非标准 IoU |
| 类别抑制 | 仅同类之间抑制（符合 YOLO 惯例） |
| 候选框数量上限 | `bbox_max = 输入缓冲字节数 / sizeof(BBOX)`，416 输入约 **10.4 万** 个 |
| 坐标还原 | NMS 内按 letterbox 比例 `s1/s2` 乘回原图尺度 |

---

## 六、容易误解的正常行为（⚪ 不是 bug）

### 6.1 `orig` 与 `opt` / `avx2` 的差异是 1e-1 量级

**看起来像 bug，实际是已知近似。**

`orig` 用 `fast_inverse_sqrt`（魔数 + 一次牛顿迭代，相对误差约 0.2%）做 BN 归一化，
逐层放大成 `1e-1` 量级的特征图绝对偏差。

**验证方法**：加 `-DKERNEL_ORIG_EXACT_RSQRT` 重编 `orig`，误差立刻降到 `1e-4` 量级。
详见 [testing.md 第五节](testing.md#五rsqrt-隔离法)。

### 6.2 mbv2 在不同构建下框数不同（962 vs 971）

不是"优化导致框数变化"，而是**该模型本来就在输出垃圾框**（第一节），
垃圾值对输入的微小扰动极度敏感。构建 B 下 `orig` 与 `opt` 都是 971 个，完全一致。

### 6.3 候选框数量远大于最终检测框数量

两个示例 cfg 都**没有写 `ignore_thresh`**，而解析器对缺失字段的处理是：

```c
parse_params(..., "ignore_thresh", ...);       /* 缺失时 strval = "" */
net->layer_list[layercur].ignore_thres = atof("");   /* atof("") = 0 */
```

于是 `confidence >= 0` **恒成立** → 每个网格 × 每个锚框都生成一个候选框：

| 模型 | 候选框总数（估算） | 最终框数 |
|---|---|---|
| yolo-fastest | `13×13×3 + 26×26×3 = 2535` | 3 |
| mbv2-face | `52×52×3 + 26×26×3 + 13×13×3 ≈ 10 647` | 962~971 |

这不是缺陷 —— `ignore_thresh` 在 darknet 里是**训练/评估**用的参数，
推理时的筛选用的是 NMS。只是别指望它做剪枝。

### 6.4 `13×13` 的层向量化收益低

`conv_1x1_avx2` 一次处理 8 个像素，宽度 13 的层只有 8 个能走向量路径，5 个走标量。

这是**不可避免的尾部**，不是实现问题。想改善只能改分块策略（见 [performance.md](performance.md) 第七节的 P1 项）。

### 6.5 `-avx2` 打印出 `conv kernel : opt (requested avx2)`

正常降级。原因见 [architecture.md 第九节](architecture.md#九内核派发与降级)：

- 编译期不含 AVX2 代码（非 x86 / 非 GCC），或
- 运行期 CPU 不支持 AVX2+FMA。

降级是**静默**的（只提示一次），程序照常运行，只是速度不同。

---

## 七、修复优先级建议

按"危害 × 触发概率"排序：

| 优先级 | 问题 | 为什么排这个位置 |
|---|---|---|
| **P0** | 第二节：未知段静默丢弃 | 换任何一个新模型都可能踩到，且症状（结果不对但无报错）最难查 |
| **P0** | 第三节：权重文件缺少校验 | 三行代码就能修，收益明确 |
| **P1** | 第一节：YOLO 解码用 `anchor_num` + 由 `filters` 推步长 | 需要同时校 cfg 数据；修完必须重录基线 |
| **P1** | 第四节：未知激活至少打印警告 | 一行提示就能省掉一整天的排查 |
| **P2** | 5.2 的 depthwise 通用向量化 | 只影响未来模型的性能 |
| **P3** | 多线程 / 量化 | 属于"另一个项目"，见 [performance.md](performance.md#七后续优化方向与优先级) |

### 修完之后必须做的验证

```bash
# 1. 三套内核一致性（构建 B 必须通过）
./t_cons_b test.bmp <cfg> <weights>

# 2. 单内核测试
./t_dw && ./t_1x1

# 3. 如果改了 YOLO 解码，用 yolo-fastest 确认没回归
./test -opt test.bmp yolo-fastest-1.1.cfg yolo-fastest-1.1.weights
#   仍应输出 3 个框，坐标与修改前一致
```

---

## 延伸阅读

- [architecture.md](architecture.md) —— 布局与不变量（很多"看起来是 bug"的行为源于此）
- [testing.md](testing.md) —— 如何区分"近似"与"实现错误"
- [kernels.md](kernels.md) —— 内核的能力边界（如 depthwise 仅 3×3）
