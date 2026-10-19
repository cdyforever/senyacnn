# 引擎内部设计

> 本文描述 senyacnn 的内存布局、缓冲分配、解析流程与前向调度。
> 如果你只想看内核实现，请跳转 [kernels.md](kernels.md)；如果想知道"为什么这样设计"，本文就是答案。

---

## 目录

- [一、整体数据流](#一整体数据流)
- [二、特征图布局：CHW + pad 边缘](#二特征图布局chw--pad-边缘)
- [三、pad 填充策略](#三pad-填充策略)
- [四、窗口锚点约定](#四窗口锚点约定)
- [五、层槽位错位一格的约定](#五层槽位错位一格的约定)
- [六、内存分配：arena 一次到位](#六内存分配arena-一次到位)
- [七、net_load 的六个阶段](#七net_load-的六个阶段)
- [八、BN 折叠](#八bn-折叠)
- [九、内核派发与降级](#九内核派发与降级)
- [十、检测框复用输入缓冲](#十检测框复用输入缓冲)
- [十一、前向主循环](#十一前向主循环)
- [十二、不变量清单](#十二不变量清单)

---

## 一、整体数据流

```
 BMP 文件
    │  bmp_load()
    ▼
 24bpp BGR 像素 ──net_input()──> letterbox 缩放 + 减均值 / 乘 norm
                                        │
                                        ▼
                              slot0 的 CHW 浮点缓冲
                                        │
              ┌─────────────────────────┘
              ▼
       net_forward()  逐层执行
              │
    ┌─────────┼──────────┬──────────┬──────────┬─────────┐
    ▼         ▼          ▼          ▼          ▼         ▼
  conv     maxpool    upsample    shortcut    route    yolo
    │         │          │          │          │         │
    └─────────┴──────────┴──────────┴──────────┴─────────┘
                                        │
                                        ▼
                            net->bbox_list（别名 slot0.data）
                                        │
                                    nms()
                                        │
                                        ▼
                              bbox_list 就地压缩 + 坐标还原
```

每一层只做一件事：读 `layer_list[i].matrix`（输入），写 `layer_list[i+1].matrix`（输出）。
因为槽位是预先按拓扑序排好的，**前向过程不需要任何动态分派表或拓扑排序**。

---

## 二、特征图布局：CHW + pad 边缘

### 2.1 平面存储

特征图按 **CHW** 排列（channel → row → col 依次连续），不用 HWC。理由：

| | CHW | HWC |
|---|---|---|
| 单个通道内的空间邻域 | **连续**（卷积窗口读取的最佳形态） | 跨步 `C` 个 float |
| 同一像素的跨通道 | 跨一个平面 | 连续（1×1 卷积的最佳形态） |
| 本项目的瓶颈 | 卷积窗口（im2col） | 1×1 有专门内核做向量化 load |

也就是说：CHW 让"卷积窗口"这一最普遍的访问模式变成连续读，代价是 1×1 卷积需要跨平面跳。
1×1 卷积的跳步是**规则**的（固定 `mw*mh` 的整数倍），可以用 `dot4` / AVX2 8 路累加器掩盖；
而窗口若跨步 `C`，im2col 拷贝会变成散点访问，代价更高且难以向量化。

### 2.2 每个通道平面都带 pad 边缘

`MATRIX` 里的 `width` / `height` 是**逻辑尺寸**，`data` 指向的缓冲实际是 `(width + 2*pad) × (height + 2*pad)`：

```c
mw = width  + 2 * pad;      // 行宽（float 数）
mh = height + 2 * pad;      // 行数
plane_size = mw * mh;       // 单通道 float 数
// 内部像素 (x, y) 的地址：
p = data + c * plane_size + y * mw + x;
```

`pad = 1` 时的内存形状：

```
             x=0    1    2    3   ...  mw-1
        +--------------------------------+
 y=0    |  0    0    0    0   ...   0    |   ← 整行 pad
 y=1    |  0    v    v    v   ...   0    |
 y=2    |  0    v    v    v   ...   0    |   ← 内部区域从 (1,1) 开始
 ...    |  0    v    v    v   ...   0    |
 y=mh-1 |  0    0    0    0   ...   0    |   ← 整行 pad
        +--------------------------------+
                ↑
           内部像素 (x,y) 的 padded 坐标就是 (x,y)
           不加偏移 —— 这就是"内部坐标 = padded 坐标"
```

**把 pad 做进缓冲区本身**（而不是在卷积里判断边界）是这份代码最重要的一个决定，收益：

1. 卷积内核的内层循环**没有边界分支**，可以完整展开与向量化；
2. 向量内核可以一次 load 8 个 float 而不用担心越界（只要落在平面内）；
3. 边缘的值只需在加载时填一次，之后跨帧有效。

代价是**内存放大**：每个平面上多 `2*pad*(w+h) + 4*pad²` 个 float。对 3×3 卷积网络
（`pad = 1`）在小特征图上的相对开销最大 —— 例如 13×13 的平面，每个通道由 169 个 float
变成 225 个（+33%）。对总耗时的影响主要来自 cache 压力，实测仍在噪声范围内。

---

## 三、pad 填充策略

边缘值由 `matrix_fill_pad()` 在 `net_load` 阶段填一次，**之后永不修改**：

```c
void matrix_fill_pad(MATRIX *mat, float val)
{
    for (每个通道) {
        for (y = 0; y < pad; y++)              /* 上下各 pad 行整行填 */
            for (x = 0; x < mw; x++)
                data[y*mw + x] = data[(mh-1-y)*mw + x] = val;
        for (y = pad; y < mh - pad; y++)       /* 中间行只填左右各 pad 列 */
            for (x = 0; x < pad; x++)
                data[y*mw + x] = data[y*mw + (mw-1-x)] = val;
    }
}
```

### 3.1 填什么值

| 层类型 | 缓冲区的 pad 边缘填 | 原因 |
|---|---|---|
| 普通层（conv/upsample/…） | `0` | 与"零填充"语义一致 |
| **紧接 maxpool 的输入** | `-FLT_MAX` | 取 max 时自动被忽略，池化内核不必写边界判断 |

第 306–314 行的判定是"**按槽位自己的层类型**决定填什么"，因为 `layer_list[i].matrix`
存的是"第 i 层读它时的输入 layout"，其 pad 宽度恰好等于第 i 层作为消费者需要的 pad。
所以槽位 `i` 填什么，取决于 `layer_list[i].type == LAYER_TYPE_MAXPOOL`。

> `-FLT_MAX` 而不是 `-INFINITY`：`-FLT_MAX` 参与任何有限的加减都不会产生 NaN，
> 而 `-INFINITY` 与 0 相加会得到 NaN（虽然本引擎不会让 pad 参与运算，但用有限值更安全）。

### 3.2 由此推出的一条铁律

> **所有内核只写内部区域，绝不触碰 pad 边缘。**

因为边缘值属于"下一层读取它时的语义"，本层无权改写。若某个内核顺手把 pad 区清零，
那么下一层的 maxpool 会拿到 `0` 而不是 `-FLT_MAX`，边界像素的最大值就会算错 ——
而且第一帧看不出来（缓冲刚填好），**跨帧累积**后才暴露。这类 bug 极难定位，
所以三套内核的实现里都对写入范围做了显式约束。

---

## 四、窗口锚点约定

这是历史上真实踩过的坑，三套内核必须完全一致。

对 `size × size`、`stride` 的卷积窗口，输出像素 `(ox, oy)` 覆盖输入 padded 数组的：

```
padded[oy*stride + fy][ox*stride + fx]     fy, fx ∈ [0, size)
```

**锚点直接用 padded 坐标，不加 `+pad` 修正。**

### 为什么不需要修正

因为 pad 已经物理包含在数组里：内部像素 `(ix, iy)` 的 padded 坐标**恰好**就是 `(ix, iy)`。
展开看就是：

```
输出 (0,0) 的窗口 = padded[0..size-1][0..size-1]
                  = 左上角正好落在原有的 pad 区左上角
```

这等价于"以内部像素 `(0,0)` 为中心、半径 `size/2` 的窗口"，与 darknet 语义一致。

### 如果误加 +pad 会怎样

```
错误做法：padded[oy*stride + fy + pad][ox*stride + fx + pad]
```

1. 整个输出**平移一个像素**（肉眼不易发现，但检测框会系统性偏移）；
2. 右/下边界越界读 —— 最后一个输出像素的窗口尾部会超出平面；
3. im2col / depthwise / 1×1 三条路径若有一处漏改，三者结果就会互不一致。

因此 `kernel_im2row()`、`conv_depthwise*`、`conv_1x1*` 全部遵守同一约定，
并由 `test_dw.c` / `test_1x1.c` 用**独立写的朴素三重循环**逐元素校验。

### 输出尺寸公式

```c
out_w = (in_w - size + 2*pad) / stride + 1;   // conv
out_w =  in_w / stride;                        // 池化类
```

`pad` 的来源是 cfg 里的 `pad` 标志展开成 `size / 2`（darknet 语义，仅对奇数核对称）：

```c
in->matrix.pad = in->matrix.pad ? in->filter.size / 2 : 0;
```

---

## 五、层槽位错位一格的约定

`layer_list[i]` 描述**第 i 层的参数**，但它的**输出尺寸**写在 `layer_list[i+1].matrix` 里：

```c
ilayer = net->layer_list + i + 0;   // 本层参数：type / filter / depend_list
olayer = net->layer_list + i + 1;   // 本层的输出矩阵（也就是下一层的输入）
```

```
 layer_list:  [0]      [1]      [2]      [3]      ...   [N]
              │        │        │        │              │
 matrix:      │        │        │        │              │
 语义:      输入buf   层0出    层1出    层2出   ...    层N-1出
               ▲        ▲        ▲
               │        │        │
             net_input  层0      层1
                       写这里    写这里
```

那么 `layer_list[0].matrix` 是什么？是**网络输入缓冲**（`net_input` 写入的地方），
由 slot0 的 `calloc` 单独分配。

这样设计的好处：

- 一个槽位同时被"上一层当输出"和"本层当输入"使用，**尺寸只存一份**，不可能不一致；
- 每层只需处理 `(ilayer, olayer)` 一对指针，签名统一；
- `route` / `shortcut` 只需存源层下标，取 `layer_list[idx + 1].matrix` 即得被引用的输出。

---

## 六、内存分配：arena 一次到位

`net_forward` 里**没有任何 `malloc` / `free`**。所有缓冲在 `net_load` 中一次分好：

| 缓冲 | 内容 | 生命周期 |
|---|---|---|
| `weight_buf` | darknet `.weights` 原始内容（float 数组） | 全程 |
| `foldbuf` | BN 折叠后的 `W'`/`b'` 副本片区（按所有 BN 层规模预留） | 全程 |
| `layerbuf` | **所有层输出**的 arena，按槽位切片 | 全程 |
| `cnntempbuf` | 卷积临时缓冲（im2col 行块），按最大需求一次到位 | 全程 |
| `layer_list[0].matrix.data` | 网络输入缓冲 + **检测框存储** | 全程 |

```
 weight_buf  ┌──────────────────────────────────────┐  原始权重，只读
             │ w0 │ w1 │ w2 │ ... │ wN                │
             └──────────────────────────────────────┘
 foldbuf     ┌──────────────────────────────────────┐  BN 层才有，opt/avx2 读这里
             │ W'0 │ b'0 │ W'1 │ b'1 │ ...          │
             └──────────────────────────────────────┘
 layerbuf    ┌──────────────────────────────────────┐  m->data 都指向这里
             │slot1│slot2│slot3│ ... │slotN          │
             └──────────────────────────────────────┘
                ▲     ▲     ▲
                │     │     └─ layer_list[3].matrix.data
                │     └─────── layer_list[2].matrix.data
                └───────────── layer_list[1].matrix.data

 slot0.data  ┌──────────────────────────────────────┐
             │  输入图像 CHW（含 pad）  │ 检测框区    │  同一块内存两种用途
             └──────────────────────────────────────┘
```

### 6.1 为什么不用每层 malloc

1. **分配次数 = 层数** 时，`malloc` 的开销在 `net_forward` 里会变成可测的固定成本，
   而且会打断缓存局部性（每层拿到不同地址）；
2. arena 让所有层输出在**同一段连续地址**里，相邻层天然靠近，对 TLB 与 L2 友好；
3. 出错处理简单：`net_load` 结束要么全部分配成功，要么 `net_free` 一次性回收。

### 6.2 arena 的容量计算

```c
for (i = 1; i <= layer_num; i++) {          /* 注意从 1 开始：slot0 单独分配 */
    size    = (m->width + 2*m->pad) * (m->height + 2*m->pad) * m->channels;
    total  += size;                          /* 累加得到 layerbuf 总容量 */
    tmpmax  = max(tmpmax, im2col 需求);      /* 取最大临时缓冲需求 */
}
```

第二次遍历才把 `m->data` 切到 `layerbuf + off` 上，并顺便填 pad。

> **临时缓冲取最大值而不是求和**：`cnntempbuf` 是"当前层用完即弃"的，不需要每层独立空间。

---

## 七、net_load 的六个阶段

```
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. 解析 cfg                                                 │
 │    pipe_cfg() 逐行切分 → parse_params() 提取 key=value      │
 │    建 layer_list[]，顺推每层输出尺寸（matrix_allocate 逻辑）  │
 │    pad 标志展开为 size/2                                     │
 ├─────────────────────────────────────────────────────────────┤
 │ 2. 统计规模                                                 │
 │    累计 conv 层权重元素数 → weight_size                      │
 │    累计 BN 层需要 W'+b' 的元素数 → foldbuf 容量               │
 ├─────────────────────────────────────────────────────────────┤
 │ 3. 分配 + 读权重                                             │
 │    malloc(weight_buf / foldbuf)                             │
 │    fseek 跳过 20 字节 WEIGHTS_FILE_HEADER，fread 一遍         │
 │    同时把每个 BN 层的 data2/bias2 指向 foldbuf 内的预留片区   │
 │    （无 BN 的层 data2 = data, bias2 = bias，直接别名）        │
 ├─────────────────────────────────────────────────────────────┤
 │ 4. 输入槽位                                                 │
 │    slot0.data = calloc(...)  ← calloc 而非 malloc             │
 │    原因：letterbox 留白区与 pad 边缘都依赖"未写即为 0"        │
 ├─────────────────────────────────────────────────────────────┤
 │ 5. arena 切片 + 填 pad                                       │
 │    两遍遍历：先算总容量，再切地址并 matrix_fill_pad()         │
 ├─────────────────────────────────────────────────────────────┤
 │ 6. 反序列化锚框                                              │
 │    [yolo] 的 anchors 行按 mask 筛选，写入 anchor_list[][]     │
 │    （注意：解码阶段实际未使用 anchor_num，见 known-issues）    │
 └─────────────────────────────────────────────────────────────┘
```

`WEIGHTS_FILE_HEADER` 定义为 `#pragma pack(1)` 的 **20 字节**结构（3 个 `int32` 版本号 = 12B，
加 1 个 `uint64` 训练图数 = 8B），与 darknet 的 `.weights` 格式一致。读取时整体跳过即可 ——
引擎不关心版本号。

可以用文件大小反推验证：

| 模型 | 权重元素数 | 期望文件大小 | 实际文件大小 |
|---|---|---|---|
| yolo-fastest-1.1 | 346 062 | `346062×4+20 = 1 384 268` | 1 384 268 ✓ |
| mbv2_yolov3_face | 2 063 232 | `2063232×4+20 = 8 252 948` | 8 252 948 ✓ |

---

## 八、BN 折叠

### 8.1 三个版本的做法

| 实现 | 做法 | 位置 |
|---|---|---|
| `orig` | **每输出元素**现场算 `(dot - mean) * rsqrt(var+eps) * scale + bias`，`rsqrt` 用魔数近似 | 卷积内层循环 |
| `opt` / `avx2` | **加载时**折叠进权重副本，内层循环完全没有 BN 运算 | `net_bn_fold()` |

### 8.2 推导

原始表达式（对输出通道 `j`）：

```
out_j = act( (dot_j - mean_j) * invvar_j * scale_j + bias_j )
其中 invvar_j = 1 / sqrt(var_j + eps),  eps = 1e-5
```

令 `a_j = scale_j / sqrt(var_j + eps)`，则：

```
out_j = act( dot_j * a_j  -  mean_j * a_j + bias_j )
      = act( dot'_j + b'_j )

其中   W'_jk = W_jk * a_j        （权重整行乘一个标量）
        b'_j = bias_j - mean_j * a_j
       dot'_j = Σ_k W'_jk · x_k
```

**BN 的仿射变换被吸收进了权重和偏置**，推理期一分钱不花。

### 8.3 为什么只写 `data2` / `bias2` 而不覆盖 `data` / `bias`

```c
f->bias2[j] = f->bias[j] - f->rolling_mean[j] * a;
f->data2[j*ft + k] = f->data[j*ft + k] * a;
```

保留原始权重的原因：

1. 运行期可以**来回切换** `orig` ↔ `opt` ↔ `avx2`，而 `orig` 必须读原始权重；
2. 折叠是有损的（乘一个标量），不可逆，覆盖后就再也回不到逐位基线；
3. `test_consistency.c` 需要在同一个 `NET` 上跑三套实现做比对。

无 BN 的层让 `data2 = data`、`bias2 = bias`（**别名，不复制内存**），
这样三条内核路径都可以统一读 `data2/bias2`，不需要在内层判断 `batchnorm`。

### 8.4 eps 与数值差异

```c
float a = f->scale[j] / sqrtf(f->rolling_variance[j] + 0.00001f);
```

- 用 **`sqrtf`（精确）**，而不是 `orig` 的 `kernel_fast_inverse_sqrt`（魔数近似，约 0.2% 相对误差）；
- 这就是 `orig` 与 `opt` 存在 1e-1 量级差异的**唯一原因**，与内核正确性无关。
  用 `-DKERNEL_ORIG_EXACT_RSQRT` 重编 `orig` 即可把这项隔离掉，详见 [testing.md](testing.md)。

### 8.5 折叠的触发与幂等

```c
// net_forward 开头
if (!net->bn_fused && kernel_impl_needs_bn_fold(net->conv_impl)) net_bn_fold(net);

// net_set_conv_impl 里
net->bn_fused = 0;   // 切换实现时让状态重新判定
```

| 情形 | `bn_fused` | 动作 |
|---|---|---|
| 首次跑 opt/avx2 | 0 | 折叠，置 1 |
| 继续跑 opt/avx2 | 1 | 跳过 |
| 切回 orig | 0（被重置） | `kernel_impl_needs_bn_fold(ORIG)` 返回 0，不折叠 |
| 从 orig 切到 avx2 | 0 | 折叠 |

`net_bn_fold` 自身也幂等（开头 `if (net->bn_fused) return;`），重复调用结果相同。
折叠只读原始参数，因此**不修改任何被 `orig` 依赖的数据**。

---

## 九、内核派发与降级

`net_forward` **只调用一个入口** `kernel_conv_forward()`：

```c
void kernel_conv_forward(NET *net, LAYER *ilayer, LAYER *olayer)
{
    switch (net->conv_impl) {
    case CONV_IMPL_AVX2:
        if (kernel_avx2_compiled() && kernel_avx2_supported()) {
            kernel_conv_forward_avx2(net, ilayer, olayer);
            return;
        }
        /* 降级：只提示一次，避免每帧刷屏 */
        warn_once();
        /* fall through */
    case CONV_IMPL_OPT:
        kernel_conv_forward_opt(net, ilayer, olayer);
        return;
    default:
        kernel_conv_forward_orig(net, ilayer, olayer);
    }
}
```

### 9.1 两个独立的条件

| 检查 | 时机 | 判据 | 结果 |
|---|---|---|---|
| `kernel_avx2_compiled()` | 编译期 → 运行期常量 | `#if KERNEL_HAVE_AVX2` | 本产物是否含 AVX2 代码 |
| `kernel_avx2_supported()` | 运行期，结果缓存 | `__builtin_cpu_supports("avx2") && ("fma")` | 本 CPU 能否执行 |

**分开的原因**：同一份产物要能跨机器分发。构建机有 AVX2 不代表目标机有；

反过来，在非 x86 上编译时 `KERNEL_HAVE_AVX2 = 0`，AVX2 源文件的函数体整体被
`#if` 排除，只剩一个空壳，链接不会失败。

### 9.2 指令集是怎么进来的

```c
#pragma GCC push_options
#pragma GCC target("avx2,fma")     // ← 只作用于本文件
   ... 所有 AVX2 内核 ...
#pragma GCC pop_options
```

不用全局 `-march=native` 的理由：

1. 宿主工程的编译选项不必改，其它文件的代码生成不受影响；
2. 不会把整个程序编译成只能在构建机上跑的指令集（否则在老 CPU 上直接
   `SIGILL`，连降级的机会都没有）；
3. 实测在 Zen 4 上 `-march=native` 并不比 `-mavx2 -mfma` 快（见 [performance.md](performance.md)）。

### 9.3 状态机

```
        net_set_conv_impl(net, X)
                 │
                 ├─ X == AVX2 且 !compiled()  →  拒绝，返回 -1，旧实现保持
                 └─ 否则 net->conv_impl = X; net->bn_fused = 0
                                │
                                ▼
                    ┌───────────────────────┐
                    │  net_effective_conv_impl()  │
                    └───────────────────────┘
                                │
     AVX2 且 compiled && supported ──> 实际跑 avx2
     AVX2 但 CPU 不支持          ──> 实际跑 opt（打印 requested avx2）
     ORIG / OPT                 ──> 原样
```

`main.c` 打印的 `conv kernel : avx2 (requested avx2)` 中，前者就是
`net_effective_conv_impl()` 的结果，后者是命令行请求 —— 两者不同即说明发生了降级。

---

## 十、检测框复用输入缓冲

```c
// parser.c
mat->data     = calloc(1, (w + 2*pad) * (h + 2*pad) * channels * sizeof(float));
net->bbox_max = (w + 2*pad) * (h + 2*pad) * channels * sizeof(float) / sizeof(BBOX);
net->bbox_list = (BBOX*)mat->data;
```

把输入图像缓冲**当成 `BBOX` 数组**来用：

| | 大小 |
|---|---|
| 槽位 0 缓冲 | `320×320×3 × 4B = 1.23 MB` |
| 可容纳 `BBOX` 数 | `1.23 MB / 20B ≈ 61440` 个 |

为什么可以这样：

1. **生命周期不重叠**：`net_input` 在 `net_forward` 最开始把图像写进这块内存；
   `layer_yolo_forward` 在最后阶段才往里写检测框。中间所有层已经读过输入并写完了自己的输出槽位；
2. `BBOX` 是纯 POD（`int` + 4 个 `float`，20 字节），无构造/析构语义；
3. 省掉一次随图像尺寸变化的分配。

风险与防护：

- **`bbox_num` 必须有上限检查**：`if (net->bbox_num < net->bbox_max)`。
  没有上限时，一个解码 bug（例如 [known-issues](known-issues.md) 里的 mbv2 情形）
  会直接写穿整个堆；
- 只在一次前向的**尾部**使用，中间层若有任何代码重新读 slot0 就会拿到垃圾值。

> 这是一个"省一次分配"换取"必须小心"的取舍。如果你要在 slot0 上做新用途，先确认时序。

---

## 十一、前向主循环

```c
void net_forward(NET *net)
{
    /* 1. 按需折叠 BN（opt/avx2 只认折叠后的副本） */
    if (!net->bn_fused && kernel_impl_needs_bn_fold(net->conv_impl)) net_bn_fold(net);

    /* 2. 统计 route/shortcut 的引用计数（仅 net_dump 展示用） */
    ...

    /* 3. 逐层执行：ilayer 是参数，olayer = ilayer+1 是输出矩阵 */
    for (i = 0; i < net->layer_num; i++) {
        ilayer = net->layer_list + i + 0;
        olayer = net->layer_list + i + 1;
        switch (ilayer->type) {
        case LAYER_TYPE_CONV    : kernel_conv_forward(net, ilayer, olayer);    break;
        case LAYER_TYPE_AVGPOOL : layer_avgmaxpool_forward(ilayer, olayer, 0); break;
        case LAYER_TYPE_MAXPOOL : layer_avgmaxpool_forward(ilayer, olayer, 1); break;
        case LAYER_TYPE_UPSAMPLE: layer_upsample_forward  (ilayer, olayer);    break;
        case LAYER_TYPE_DROPOUT : layer_dropout_forward   (ilayer, olayer);    break;
        case LAYER_TYPE_SHORTCUT: layer_shortcut_forward  (net, ilayer, olayer); break;
        case LAYER_TYPE_ROUTE   : layer_route_forward     (net, ilayer, olayer); break;
        case LAYER_TYPE_YOLO    : layer_yolo_forward      (net, ilayer);       break;
        }
    }

    /* 4. NMS：就地压缩 bbox_list，并把坐标从网络输入尺度还原到原图尺度 */
    net->bbox_num = nms(net->bbox_list, net->bbox_num, 0.5f, 1, net->s1, net->s2);
}
```

注意 `LAYER_TYPE_YOLO` 的调用**不需要 `olayer`** —— yolo 层不产生输出矩阵，
它直接把候选框写进 `net->bbox_list`。

### 特殊层的输出槽位处理

| 层 | 输出槽位怎么来 |
|---|---|
| `dropout` | 推理期恒等映射。**输出槽位直接别名输入缓冲**（`olayer->matrix.data = ilayer->matrix.data`），不占 arena 空间 |
| `shortcut` | 读 `depend_list[0]+1` 槽位，与输入逐元素相加，写入输出槽位 |
| `route` | 按 `depend_list` 把多路通道维拼接（最多 4 路），写入输出槽位。yolo-fastest 里用于 FPN 特征融合 |

---

## 十二、不变量清单

改这份代码时请逐条确认。破坏任意一条都会引起**难以定位**的错误。

| # | 不变量 | 破坏后果 |
|---|---|---|
| 1 | 特征图是 CHW 平面存储 | 所有内核失效 |
| 2 | 平面实际尺寸为 `(w+2p) × (h+2p)`，内部像素 `(x,y)` 地址 = `data + y*mw + x` | 整体错位 |
| 3 | 内核**只写内部区域**，绝不碰 pad 边缘 | 跨帧累积错值（首帧正常，越跑越偏） |
| 4 | 卷积窗口锚点用 padded 坐标，**不加 `+pad`** | 输出平移一个像素 + 右/下越界 |
| 5 | `layer_list[i+1].matrix` 是第 `i` 层的输出 | 层间数据错乱 |
| 6 | `layer_list[0].matrix` 是网络输入，由 `calloc` 分配 | letterbox 留白区不是 0 |
| 7 | `net_forward` 内零 `malloc` | 性能退化 + 内存管理复杂化 |
| 8 | `orig` 只读 `data`/`bias`/`scale`/`mean`/`var`；`opt`/`avx2` 只读 `data2`/`bias2` | 折叠状态与实际读取不一致 → 结果错 |
| 9 | 写 `bbox_list` 前必须检查 `bbox_num < bbox_max` | 堆溢出 |
| 10 | 切换实现必须清 `bn_fused` | 用未折叠的副本算 BN → 结果错 |

---

## 延伸阅读

- [kernels.md](kernels.md) —— 三套内核的具体算法与实现细节
- [performance.md](performance.md) —— 这些设计付出的性能代价与实测数据
- [testing.md](testing.md) —— 上述不变量是怎么被测试覆盖的
- [known-issues.md](known-issues.md) —— 已知未修的问题
