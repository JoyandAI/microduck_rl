#!/usr/bin/env bash
# 7.5 V 最终重拟合（15 cm 标签换位修正 + 150 g rerun2 替换）: m1..m6 并行 + 独立评估
#
#   bash scripts/fit_hd1910_75v_v2.sh            # 默认 20000 trials
#   TRIALS=2000 bash scripts/fit_hd1910_75v_v2.sh
#
# 产物：hd1910_calibration/fit75v2/fit_m{1..6}_75v2.json + mae_report.json + DONE_*
set -u
cd "$(dirname "$0")/.."

OUT=hd1910_calibration/fit75v2
TRAIN=hd1910_calibration/bench75_train48_v2
VAL=hd1910_calibration/bench75_val24_v2
# 收敛实测(2026-09-15): m1 ~1000 trial 到位(之后 9000 次只改第 10 位小数), m3/m5/m6 3000-5000 基本平
# ⇒ 默认 5000 足够; 并列判据精度 0.0005 rad 远大于后期改进量。要更保险可 TRIALS=20000 覆盖。
TRIALS="${TRIALS:-5000}"
SET="{'kt': 0.692, 'R': 3.75, 'error_gain': 0.163, 'kd': 0.346, 'max_current': 3.25, 'max_velocity': 12.0423, 'max_acceleration': 500.0}"

mkdir -p "$OUT" logs
rm -f "$OUT"/DONE_* "$OUT"/FIT_FAILED

echo "[i] $(date '+%F %T') 开始: trials=$TRIALS train=$(ls $TRAIN/*.json | wc -l) val=$(ls $VAL/*.json | wc -l)"
for M in m1 m2 m3 m4 m5 m6; do
  (
    PYTHONPATH=vendor/bam .venv/bin/python -m bam.fit \
      --logdir "$TRAIN" --actuator hd1910 --model "$M" \
      --method cmaes --trials "$TRIALS" --workers 1 --set "$SET" \
      --output "$OUT/fit_${M}_75v2.json" > "logs/fit75v2_${M}.log" 2>&1 \
      && touch "$OUT/DONE_$M" || echo "$M" >> "$OUT/FIT_FAILED"
  ) &
done
wait

if [ -f "$OUT/FIT_FAILED" ]; then
  echo "[!] 拟合失败: $(tr '\n' ' ' < "$OUT/FIT_FAILED")"
else
  echo "[i] $(date '+%F %T') 六档拟合完成，开始独立评估"
  PYTHONPATH=vendor/bam .venv/bin/python scripts/eval_bam_rollout.py \
    --fit-dir "$OUT" --train-dir "$TRAIN" --val-dir "$VAL" \
    > logs/fit75v2_eval.log 2>&1 && touch "$OUT/DONE_EVAL"
fi
touch "$OUT/DONE_ALL"
echo "[i] $(date '+%F %T') 全部结束"
