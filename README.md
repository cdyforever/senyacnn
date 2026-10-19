# senyacnn

一个用纯 C 写的**轻量级 CNN 推理引擎**：直接解析 darknet 的 `.cfg` / `.weights`，
不依赖任何深度学习框架，可用于 YOLO 系列目标检测模型的嵌入式/桌面端推理。

代码从零实现，只依赖 libc 与 libm。卷积部分提供**三套可运行期切换的实现**：
原始标量、优化标量、AVX2 向量化，便于做正确性对拍与性能 A/B。

```
输入图像 ──letterbox──> CHW 特征图 ──卷积/池化/上采样/shortcut/route──> YOLO 解码 ──NMS──> 检测框
```

---

## 目录

- [为什么值得看](#为什么值得看)
- [快速开始](#快速开始)
- [三套卷积内核](#三套卷积内核)
- [性能](#性能)
- [项目结构](#项目结构)
- [支持的层类型](#支持的层类型)
- [文档](#文档)
- [测试](#测试)
- [已知问题](#已知问题)

---

## 为什么值得看

这是一个"把推理引擎拆开看清楚"的教学式实现，而不是追求极致性能的框架。它刻意保留了三套并存的内核实现，因此可以：

- **看见每一步优化值多少**：从单链累加到 4 路累加器、从通用 im2col 到 depthwise/1x1 特化、再到 AVX2，每一层改动都有可复现的对比数据。
- **验证优化没有改坏结果**：`test_consistency.c` 做逐层特征图比对，把"浮点顺序差异"和"实现错误"分开度量，并专门区分了对拍基准自身的近似误差。
- **读懂内存布局的代价**：特征图带 pad 边缘、"内部坐标 = padded 坐标"的约定、arena 一次性分配、BN 加载时折叠，这些设计都在注释里解释了原因。
- **量力而行地优化**：文档里记录了哪些优化**试过但没用**（例如 8 路输出通道分块、depthwise 激活外提），避免后人重走弯路。

## 快速开始

### 依赖

- GCC / MinGW-w64（Windows）或 cc（Linux）
- 无需 OpenCV、无需 BLAS，只要 libc + libm
- AVX2 内核需要编译器支持函数级 `#pragma GCC target`（GCC / Clang）；没有也能编译，运行期自动降级

### 构建

```bash
# Linux
sh build-linux.sh

# Windows (MinGW / MSYS)
sh build-win.sh
```

两条脚本等价于：

```bash
gcc -Wall -O3 \
    bmpfile.c parser.c layer_conv.c layer_pool.c layer_misc.c layer_yolo.c nms.c net.c main.c \
    kernels/common/kernel_shared.c \
    kernels/common/kernel_common.c \
    kernels/orig/kernel_conv_orig.c \
    kernels/opt/kernel_conv_opt.c \
    kernels/avx2/kernel_conv_avx2.c \
    -o test -lm
```

> **不需要加 `-march=native`。** AVX2 内核用文件级 `#pragma GCC target("avx2,fma")` 指定指令集，
> 所以产物在任意 x86-64 上都能加载，能否真正执行由运行期 CPU 检测决定。

### 运行

```bash
# 默认模型：yolo-fastest-1.1（320x320，80 类）
./test

# 指定图片 / 模型
./test test.bmp yolo-fastest-1.1.cfg yolo-fastest-1.1.weights
./test test.bmp mbv2_yolov3_face.cfg mbv2_yolov3_face_final.weights

# 选择卷积内核
./test                # orig  原始标量（逐位可复现的基准）
./test -opt           # opt   优化标量
./test -avx2          # avx2  AVX2 + FMA 向量化
./test -impl=avx2     # 同上，显式写名字（大小写不敏感）

# 调整计时帧数（默认 10 帧，输出总毫秒数）
./test -avx2 -n=50
```

输出包含网络逐层结构表、推理耗时、检测框列表，并把框画在 `out.bmp` 上：

```
conv kernel : avx2 (requested avx2)
layer   type  filters fltsize  pad/strd input          output       bn/act  ref
  0 conv       8/  1  3x 3x  3   1/ 1   320x320x  3 -> 160x160x  8  1/leaky  0
  ...
125
score: 0.99, category:  0, rect: (195  96 279 374)
```

## 三套卷积内核

三套实现的函数签名完全一致（`NET*, LAYER*, LAYER*`），由 `net->conv_impl` 在运行期切换。

| 目录 | 实现 | 关键手段 | 定位 |
|---|---|---|---|
| `kernels/orig` | 原始标量 | 单链累加 + 每元素重算 BN（含 `fast_inverse_sqrt` 近似） | **逐位可复现的基准**，用于回归对拍 |
| `kernels/opt` | 优化标量 | BN 加载时折叠、按几何分派 depthwise/1x1/通用、`dot4` 四路累加、按行连续写 | 无 CPU 扩展依赖，可移植性最好 |
| `kernels/avx2` | AVX2 + FMA | 8 像素/向量、权重常驻寄存器、4 输出通道共享一次 load、8 路点积 | x86 上最快，CPU 不支持时自动降级 |

三者关系：

```
                 ┌──────────────────┐
                 │  kernel_conv_forward(net,…)   ← net_forward 只调用这一个入口
                 └──────────┬───────┘
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   CONV_IMPL_ORIG      CONV_IMPL_OPT      CONV_IMPL_AVX2
   kernel_conv_        kernel_conv_       kernel_conv_
    forward_orig        forward_opt        forward_avx2
        │                   │                   │
        └───── 共用 kernels/kernel_common.h 的布局与窗口锚点约定 ─────┘
```

`kernels/common/` 存放三套共用的东西：`activate` / `im2row` / CPU 能力检测（`kernel_shared.c`），
以及选路与降级逻辑（`kernel_common.c`）。

## 性能

测试环境：AMD Ryzen 7 8745HS（Zen 4，8C/16T），GCC 16.2，`-Wall -O3`，单线程，连续 10 帧取总耗时。

| 模型 | 输入 | orig | opt | avx2 | opt 加速 | avx2 加速 |
|---|---|---|---|---|---|---|
| yolo-fastest-1.1 | 320×320 | 1046 ms | 453 ms | **125 ms** | 2.31× | **8.37×** |
| mbv2_yolov3_face | 416×416 | 7891 ms | 2875 ms | **1203 ms** | 2.74× | **6.56×** |

按内核类型拆分的耗时占比（opt 路径）：

| 模型 | depthwise | 1×1 | 通用 im2col |
|---|---|---|---|
| yolo-fastest | 36.7 % | 59.0 % | 2.5 % |
| mbv2-face | 13.7 % | 77.8 % | 7.3 % |

> 1×1 卷积是主要瓶颈（MobileNet 系网络的算力主体），因此 AVX2 版的 `conv_1x1_avx2`
> 做了"4 个输出通道共享一次输入 load"，是收益最大的一处改动。详见 [docs/performance.md](docs/performance.md)。

## 项目结构

```
senyacnn/
├── README.md
├── docs/                        设计、内核、性能、测试、已知问题文档
│
├── main.c                       演示入口：命令行解析、计时、画框
├── net.c                        网络生命周期：pad 填充 / 释放 / 输入预处理 / 前向调度 / 结构打印
├── parser.c                     cfg 解析 + weights 装载 + 缓冲预分配 + BN 折叠
├── nms.c                        非极大值抑制
├── bmpfile.c / bmpfile.h        BMP 读写（含矩形绘制）
│
├── layer_conv.c                 卷积入口（只转发到 kernels/，保留历史符号）
├── layer_pool.c                 avgpool / maxpool / upsample
├── layer_misc.c                 dropout / shortcut / route
├── layer_yolo.c                 YOLO 检测头解码
│
├── kernels/
│   ├── kernel_common.h          三套内核共享的接口与不变量（布局、pad、窗口锚点）
│   ├── common/
│   │   ├── kernel_shared.c      activate / im2row / CPU 能力检测
│   │   └── kernel_common.c      运行期派发 + 自动降级
│   ├── orig/kernel_conv_orig.c  原始标量实现
│   ├── opt/kernel_conv_opt.c    优化标量实现
│   └── avx2/kernel_conv_avx2.c  AVX2 + FMA 实现
│
├── test_dw.c                    conv_depthwise 单元测试（对拍朴素实现）
├── test_1x1.c                   conv_1x1 单元测试（对拍三重循环）
├── test_consistency.c           三套内核一致性测试（逐层特征图 + 检测框 + 确定性）
│
├── build-win.sh / build-linux.sh
├── compare.sh                   orig vs opt/avx2 的耗时、输出、图片指纹对比 + 一致性关卡
│
├── yolo-fastest-1.1.cfg / .weights        320×320，131 层，80 类
├── mbv2_yolov3_face.cfg / .weights        416×416，107 层，1 类
└── test.bmp                               示例输入图
```

模型规模参考：

| 模型 | 层数 | conv | maxpool | upsample | dropout | shortcut | route | yolo | 权重元素 | 权重文件 |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo-fastest-1.1 | 131 | 84 | 3 | 1 | 18 | 18 | 5 | 2 | 346 062 | 1.38 MB |
| mbv2_yolov3_face | 107 | 77 | 0 | 2 | 0 | 12 | 13 | 3 | 2 063 232 | 8.25 MB |

## 支持的层类型

| cfg 段名 | 内部枚举 | 说明 |
|---|---|---|
| `[net]` | — | 定义输入 `width` / `height` / `channels` |
| `[conv]` `[convolutional]` | `LAYER_TYPE_CONV` | 支持 `filters` `size` `stride` `groups` `pad` `batch_normalize` `activation`；`groups == channels` 时即 depthwise |
| `[maxpool]` `[max]` | `LAYER_TYPE_MAXPOOL` | `size` / `stride`；输入 pad 边缘填 `-FLT_MAX`，取 max 时自动忽略越界 |
| `[avgpool]` `[avg]` | `LAYER_TYPE_AVGPOOL` | 同上，除数为 `size²`（与 darknet 一致） |
| `[upsample]` | `LAYER_TYPE_UPSAMPLE` | 最近邻，`stride` 为放大倍数 |
| `[dropout]` | `LAYER_TYPE_DROPOUT` | 推理期恒等映射，输出槽位直接别名输入缓冲 |
| `[shortcut]` | `LAYER_TYPE_SHORTCUT` | 残差加法，`from` 支持负相对下标 |
| `[route]` | `LAYER_TYPE_ROUTE` | 通道维拼接，最多 4 路 |
| `[yolo]` | `LAYER_TYPE_YOLO` | 检测头解码；**注意当前实现有已知限制，见下** |

激活函数支持 `linear` / `relu` / `leaky`（`sigmoid` 仅在 YOLO 解码内部使用）。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 引擎内部设计：数据布局、pad 策略、窗口锚点约定、arena 分配、解析流程、派发机制、BN 折叠 |
| [docs/kernels.md](docs/kernels.md) | 三套卷积内核逐个剖析：算法、边界推导、为什么这么做、试过但没用的优化、如何新增第四套 |
| [docs/performance.md](docs/performance.md) | 基准数据、逐层剖析方法、热点分析、编译器选项实测、后续优化方向与优先级 |
| [docs/testing.md](docs/testing.md) | 三个测试程序各验证什么、判定阈值依据、如何用 rsqrt 开关隔离差异归因 |
| [docs/known-issues.md](docs/known-issues.md) | 已知限制与待修项（含 mbv2 解码缺陷的完整定位） |

## 测试

```bash
# 单内核正确性：对拍独立的朴素实现
gcc -O2 -I. test_dw.c  kernels/common/kernel_shared.c -lm -o t_dw  && ./t_dw
gcc -O2 -I. test_1x1.c kernels/common/kernel_shared.c -lm -o t_1x1 && ./t_1x1

# 三套内核一致性：逐层特征图 + 检测框 + 确定性
gcc -O3 test_consistency.c bmpfile.c parser.c layer_conv.c layer_pool.c \
    layer_misc.c layer_yolo.c nms.c net.c \
    kernels/common/kernel_shared.c kernels/common/kernel_common.c \
    kernels/orig/kernel_conv_orig.c kernels/opt/kernel_conv_opt.c \
    kernels/avx2/kernel_conv_avx2.c -o t_cons -lm
./t_cons test.bmp yolo-fastest-1.1.cfg yolo-fastest-1.1.weights

# 一键全流程（构建 + 单测 + 一致性 + 计时对比），Linux/WSL
sh compare.sh 3
```

实测结论（详见 [docs/testing.md](docs/testing.md)）：

| 对比 | yolo-fastest | mbv2-face | 判定 |
|---|---|---|---|
| `opt` vs `avx2` | 8.33e-05 | 1.05e-04 | 数值等价 |
| `orig` vs `opt`（orig 用近似 rsqrt） | 5.78e-01 | 1.21e+00 | 由 orig 自身的近似解释 |
| `orig` vs `opt`（加 `-DKERNEL_ORIG_EXACT_RSQRT`） | 1.04e-04 | 2.29e-04 | 数值等价 |

即：**优化不改变模型输出**，全部差异可由浮点结合顺序与 orig 的近似开方解释。
三套实现在重复运行下均逐位确定。

## 已知问题

**`mbv2_yolov3_face` 的检测框是垃圾值**（坐标可达 ±1e5）。这不是本项目引入的，
而是 `layer_yolo.c` 的解码与本 cfg 的数据布局不匹配所致，已定位到两处：

1. 锚框数被硬编码为 3：`for (k = 0; k < 3; k++)`，但该 cfg 三个头的 `mask` 分别是
   3 / 2 / 4 个锚框；
2. 解码步长按 `4 + 1 + classes` 计算，而该 cfg 写 `classes = 1`，但 `filters`
   （48 / 32 / 64）隐含每锚框 **16** 个通道，即 11 类 —— 步长应为 16 而非 6。

`yolo-fastest-1.1` 两个头的 `mask` 都是 3 个锚框、`classes = 80`、`filters = 255`
（3 × 85），完全自洽，因此它的检测结果正常。

详见 [docs/known-issues.md](docs/known-issues.md)。

---

## 致谢

解析逻辑与层语义参照 [darknet](https://github.com/pjreddie/darknet)；
示例模型为 [yolo-fastest](https://github.com/dog-qiuqiu/Yolo-Fastest) 系列与一个 MobileNetV2-YOLOv3 人脸模型。
