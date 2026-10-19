#!/bin/sh
#
# compare.sh - 原始内核(-opt 关) 与 优化内核(-opt 开) 的对比测试
#
# 内容：
#   1. 编译全部源码（gcc, 见 build-linux.sh）
#   2. 内核单元测试（test_dw / test_1x1）
#   3. 两个模型 x 两种模式各跑 N 次：统计耗时(min/avg)、检测文本差异、out.bmp 指纹
#
# 用法（Windows 下推荐经 WSL 执行）:
#   wsl sh compare.sh        # 默认每个组合跑 3 次
#   wsl sh compare.sh 5      # 自定义次数
# 或进入 WSL 后: sh compare.sh
#
# 判读标准：
#   - ORIG_*_TXT_DIFF       => 原始路径回归被破坏（必须为 0 差异）
#   - OPT_FASTEST TXT_DIFF  => yolo-fastest 上优化输出与原始不一致（目前应为 0）
#   - OPT_MBV2 TXT_DIFF=n   => mbv2 基线本身是乱框输出，仅看差异数量级是否稳定
#
N=${1:-3}

sh build-linux.sh > /dev/null 2>&1 || { echo "BUILD_FAIL"; exit 1; }
echo "BUILD_OK"

# ---------- 1. 内核单元测试 ----------
gcc -O2 -I. test_dw.c  kernels/common/kernel_shared.c -lm -o /tmp/t_dw  2>/dev/null && /tmp/t_dw  | tail -1
gcc -O2 -I. test_1x1.c kernels/common/kernel_shared.c -lm -o /tmp/t_11  2>/dev/null && /tmp/t_11

# ---------- 1b. 三套内核数值一致性（张量级逐层比对 + 确定性自检）----------
# 构建 A：orig 用 fast_inverse_sqrt 近似（历史基线行为）
# 构建 B：orig 也改用精确 sqrtf（-DKERNEL_ORIG_EXACT_RSQRT），用于隔离该近似
TC_SRC="test_consistency.c bmpfile.c parser.c layer_conv.c layer_pool.c layer_misc.c \
        layer_yolo.c nms.c net.c kernels/common/kernel_shared.c kernels/common/kernel_common.c \
        kernels/orig/kernel_conv_orig.c kernels/opt/kernel_conv_opt.c kernels/avx2/kernel_conv_avx2.c"
if gcc -O3 -I. $TC_SRC -o /tmp/t_cons_a -lm 2>/dev/null; then
  echo "--- 一致性 构建A(orig=近似rsqrt) ---"
  /tmp/t_cons_a test.bmp yolo-fastest-1.1.cfg yolo-fastest-1.1.weights | grep -E '特征图 |判定|结论'
fi
if gcc -O3 -I. -DKERNEL_ORIG_EXACT_RSQRT $TC_SRC -o /tmp/t_cons_b -lm 2>/dev/null; then
  echo "--- 一致性 构建B(orig=精确sqrtf) ---"
  /tmp/t_cons_b test.bmp yolo-fastest-1.1.cfg yolo-fastest-1.1.weights | grep -E '特征图 |判定|结论'
fi
# ---------- 2. 计时与输出采集 ----------
# run <开关> <cfg> <weights> <标签>
run() {
  i=0
  : > /tmp/cmp_$4.times
  while [ $i -lt $N ]; do
    ./test $1 test.bmp "$2" "$3" > /tmp/cmp_$4.txt 2>&1 || { echo "RUN_FAIL $4"; return 1; }
    grep -oE '[0-9]+ms' /tmp/cmp_$4.txt | tr -d 'ms' >> /tmp/cmp_$4.times
    cp out.bmp /tmp/cmp_$4.bmp
    i=$((i+1))
  done
  awk -v tag="$4" 'NR==1{min=$1}{sum+=$1}END{printf "%-14s min=%sms avg=%.0fms\n", tag, min, sum/NR}' /tmp/cmp_$4.times
}

echo "--- yolo-fastest-1.1 ---"
run ""     yolo-fastest-1.1.cfg yolo-fastest-1.1.weights       o_fastest
run "-opt" yolo-fastest-1.1.cfg yolo-fastest-1.1.weights       x_fastest
echo "--- mbv2_yolov3_face ---"
run ""     mbv2_yolov3_face.cfg mbv2_yolov3_face_final.weights o_mbv2
run "-opt" mbv2_yolov3_face.cfg mbv2_yolov3_face_final.weights x_mbv2

# ---------- 3. 正确性比对 ----------
grep -v 'ms$' /tmp/cmp_o_fastest.txt > /tmp/cmp_a.txt
grep -v 'ms$' /tmp/cmp_x_fastest.txt > /tmp/cmp_b.txt
diff /tmp/cmp_a.txt /tmp/cmp_b.txt > /tmp/cmp_d1.txt && echo "OPT_FASTEST_TXT_SAME" \
  || echo "OPT_FASTEST_TXT_DIFF(add=$(grep -c '^>' /tmp/cmp_d1.txt) del=$(grep -c '^<' /tmp/cmp_d1.txt))"

grep -v 'ms$' /tmp/cmp_o_mbv2.txt > /tmp/cmp_a.txt
grep -v 'ms$' /tmp/cmp_x_mbv2.txt > /tmp/cmp_b.txt
diff /tmp/cmp_a.txt /tmp/cmp_b.txt > /tmp/cmp_d2.txt && echo "OPT_MBV2_TXT_SAME" \
  || echo "OPT_MBV2_TXT_DIFF(add=$(grep -c '^>' /tmp/cmp_d2.txt) del=$(grep -c '^<' /tmp/cmp_d2.txt))"

echo "--- out.bmp md5 ---"
md5sum /tmp/cmp_o_fastest.bmp /tmp/cmp_x_fastest.bmp /tmp/cmp_o_mbv2.bmp /tmp/cmp_x_mbv2.bmp | awk '{print substr($1,1,8), $2}'

rm -f test out.bmp
