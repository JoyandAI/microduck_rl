# 腿摆标定 MAE 报告
- log 目录: hd1910_calibration/bench_all_processed(36 条), 拟合输出: hd1910_calibration/fit_m5
- 分组: 拟合 24 条, 独立验证 12 条(每组最后一次重复)
- 口径: bam.simulate 回放, 每 0.5s 与真机轨迹同步, 位置 MAE 平均
- 门: **独立验证 MAE < 0.157 rad**且 train/validation 无明显分叉

| 模型 | 拟合 MAE(rad) | 验证 MAE(rad) | 判定 |
|---|---:|---:|---|
| m5 | 0.03225 | 0.03209 | PASS |

## 建议
- 取 PASS 中 MAE 最小档落地到 vendor/bam/bam/params/hls2909/mN.json;
- 若全 FAIL: 检查装配(机身固定/零位/砝码 r_f)、挂载点测量, 或增大 trials;
- 外置磁编 v1(高于 150Hz 采样)可再降 MAE。
