# 走路 + 坐下起立 最终报告（2026-09-16 09:58）

队列：`scripts/train_walk_sitstand_queue.sh`；结果表 `logs/queue_walk_sitstand/results.tsv`；判定 `logs/queue_walk_sitstand/report.json`。

## 判定

```
run         rc      指纹一致        判据  说明
----------------------------------------------------------------------------------------------------
sitstand     0         ✓      PASS  起立成功
walk         0         ✓      PASS  零命令: 5 次重复全过（离地中位 0.4 / 最大 0.8 mm；偏航漂移中位 -82°，范围 -284…+26°） | 0.2m/s: 

--- 明细 ---
[sitstand]
  ok: True
  phases: {'seg0_STAND': (0.115, 35.4, -1.932), 'seg1_SIT': (0.055, 0.7, -1.932), 'seg2_STAND': (0.115, 35.4, -1.932)}
  last_stand_z_m: 0.115
  last_stand_tilt_deg: 35.4
  min_sit_z_m: 0.055
  neck_rel_min_rad: -1.932
  neck_warning: 头下垂(颈 pitch 相对 HOME -1.93 rad) —— 既有'头三脚架'问题，非本次判据
[walk]
  ok: True
  zero_cmd: {'ok': True, 'n': 5, 'per_run_ok': [True, True, True, True, True], 'foot_clearance_mm_median': 0.42183117380438506, 'foot_clearance_mm_max': 0.7547099795672113, 'tilt_max_deg_median': 2.3038624434346886, 'gait_freq_hz_median': 0.1111111111111111, 'yaw_drift_deg_median': -81.58340590832083, 'yaw_drift_deg_range': [-284.3624670634996, 26.140177829429874], 'why': '5 次重复全过（离地中位 0.4 / 最大 0.8 mm；偏航漂移中位 -82°，范围 -284…+26°）'}
  gait: {'ok': True, 'resets': 1, 'fell': False, 'tilt_max_deg': 7.160323547161372, 'gait_freq_hz': 2.3000000000000003, 'foot_clearance_mm': 28.385154131807074, 'tilt_mean_deg': 2.5703260760243456, 'yaw_drift_deg': -124.59142978085548, 'why': '步态合格'}

[i] → logs/queue_walk_sitstand/report.json
[PASS] 全部判据通过
```

## 交付件

| 文件 | 大小 | sha256 |
|---|---:|---|
| `walk_hd1910_12000.onnx` | 793733 B | `bebfc1bb40e6fb38c2b40421d691ccaf40c4bae6428eddcc3a306b806536543b` |
| `sitstand_hd1910_8000.onnx` | 793723 B | `9ef1053ffb93fd988fa7817e2c21daab8648025357da3a6cc8bb35ba7bc9a170` |

## 机器人侧替换（**未执行**，需你确认）

```bash
# 1) 备份现网策略
cp /home/joyandai/microduck/policies/alpha_walking.onnx{,.bak.$(date +%F)}
cp /home/joyandai/microduck/policies/alpha_sitstand.onnx{,.bak.$(date +%F)}
# 2) 放入新策略（保持运行时约定名，或改配置 walk = .../policies/<新名>.onnx）
cp walk_hd1910_12000.onnx /home/joyandai/microduck/policies/alpha_walking.onnx
cp sitstand_hd1910_8000.onnx /home/joyandai/microduck/policies/alpha_sitstand.onnx
# 3) 部署前彩排（CPU、忠实 BAM 环境）
.venv/bin/python scripts/infer_policy.py --walking walk_hd1910_12000.onnx
```
