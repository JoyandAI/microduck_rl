#!/usr/bin/env bash
# HD-1910 · 7.5 V 摆锤标定全网格（2026-09-15）
#
# 与 5 V 那批(2026-09-10)完全同构，只改三件事:
#   ① 舵机 ID = 23（不是 1; reg5 回读, EPROM 持久）
#   ② 输出目录加 75 电压标记 + 杆长后缀 → 绝不覆盖 5 V 的 bench_* 数据
#   ③ 供电 7.5 V（每条日志自动记录实测 vin, 拟合按实测回放, 无需其它改动）
#
# 顺序: 100g/15cm(当前已装) → 50g/15cm → 150g/15cm → 100g/10cm → 50g/10cm → 150g/10cm
# 每个质量档脚本会停下等你换砝码并输入 START(大写)。
# 换砝码前务必【先断舵机电源】; 装好后手动摆过 ±85° 复查无碰撞。
#
# 参数口径(与 5 V 批一致, 便于跨电压对比):
#   杆质量: 15cm=0.0100 kg / 10cm=0.0085 kg   (纯杆实测)
#   杆长 L: 轴心 → 砝码片质心; 150g 片更厚, 沿用 +5mm 惯例 → 15cm 档 0.155 / 10cm 档 0.105
#   近轴件: --hub-mass 0.0021 --hub-radius 0.005 (2 舵盘+8 螺丝, 实称)
#   端部质量: 50g→0.0524 / 100g→0.1024 / 150g→0.1524 (= 砝码片 + 2.4g 螺栓)
set -u
PORT=/dev/ttyACM0
ID=23                    # ⚠️ 这台是 23, 不是 1
PY=/usr/bin/python3
cd "$(dirname "$0")/.."

TRAJS=(--trajectory sin_time_square --trajectory sin_sin --trajectory lift_and_drop --trajectory up_and_down)
COMMON=(--port "$PORT" --id "$ID" --hub-mass 0.0021 --hub-radius 0.005
        --reps 3 --limit-a 0.975 --torque-budget 0.35)

echo "=== 采前只读体检(确认 reg62≈74-75 → 7.4-7.5V、mode 4、Kp32/Kd40) ==="
"$PY" /home/joyandai/bam/tools/check_servo_bench.py --port "$PORT" --id "$ID" --no-move || exit 1

# --- 0) pilot: 100g / 15cm (当前已装), 只跑 up_and_down × 1 ---
echo; echo "=== [0/6] pilot  100g / 15cm ==="
"$PY" scripts/record_pendulum_bench.py "${COMMON[@]}" \
  --tip-mass 0.1024 --arm-mass 0.0100 --arm-length 0.15 \
  --trajectory up_and_down --reps 1 --out hd1910_calibration/pilot_75

run() {  # $1=质量档标签 $2=tip $3=arm_mass $4=arm_len $5=out
  echo; echo "=== [$1] tip=$2 arm=$3/$4 → $5 ==="
  "$PY" scripts/record_pendulum_bench.py "${COMMON[@]}" \
    --tip-mass "$2" --arm-mass "$3" --arm-length "$4" \
    "${TRAJS[@]}" --out "$5"
}

# --- 15 cm 杆 ---
run "1/6  100g/15cm" 0.1024 0.0100 0.15  hd1910_calibration/bench75_wegiht100_arm15
run "2/6   50g/15cm" 0.0524 0.0100 0.15  hd1910_calibration/bench75_wegiht50_arm15
run "3/6  150g/15cm" 0.1524 0.0100 0.155 hd1910_calibration/bench75_wegiht150_arm15
# --- 10 cm 杆(换杆后重跑) ---
run "4/6  100g/10cm" 0.1024 0.0085 0.10  hd1910_calibration/bench75_wegiht100_arm10cm
run "5/6   50g/10cm" 0.0524 0.0085 0.10  hd1910_calibration/bench75_wegiht50_arm10cm
run "6/6  150g/10cm" 0.1524 0.0085 0.105 hd1910_calibration/bench75_wegiht150_arm10cm

echo; echo "=== 采集完成, 条数核对 ==="
for d in pilot_75 bench75_wegiht100_arm15 bench75_wegiht50_arm15 bench75_wegiht150_arm15 \
         bench75_wegiht100_arm10cm bench75_wegiht50_arm10cm bench75_wegiht150_arm10cm; do
  n=$(ls hd1910_calibration/$d/*.json 2>/dev/null | grep -v manifest | wc -l)
  printf "  %-34s %s 条\n" "$d" "$n"
done
echo "(15cm 三档应各 12 条, 10cm 三档应各 12 条, pilot 1 条)"

# --- 带载补测 T5(D 项) + T7(热) ---
echo; echo "=== [7/7] 带载 T5(D 项) + T7(热) ==="
"$PY" scripts/measure_hd1910_full.py --port "$PORT" --id "$ID" \
  --out hd1910_calibration/full_75
