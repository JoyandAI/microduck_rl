# m6 训练模型用途与 robotd 槽位（2026-10-03）

2026-09-27 前后在 AutoDL 上启动的四次训练。用途按启动命令、本仓库任务注册，以及上游 `pollen-robotics/microduck` 的 `deploy/robotd.toml` 对齐。本地对话记录目录当时是空的，没有额外 run 可补。

## 四次训练分别干什么

| 用途 | 机器 | 任务 | run-name | 迭代上限 | robotd 槽 | 产物 |
|---|---|---|---|---:|---|---|
| 平地行走，跌倒后自己站起来 | 4090 | `Mjlab-VelStand-Flat-MicroDuck` | `m6-VelStand-Flat` | 8000 | `walk` | `logs/rsl_rl/velstand/` 下 run 名含 `m6-VelStand-Flat` |
| 崎岖路面行走，跌倒后自己站起来 | 5090 | `Mjlab-VelStand-Rough-MicroDuck` | `m6-VelStand-Rough` | 15000 | `walk` | 同一 experiment `velstand`，和平地共用目录 |
| 从趴、躺、坐姿起立，站住后跟身体姿态。不走路 | 4090（5500 那份文件名标了 4090D） | `Mjlab-StandUp-Flat-MicroDuck` | `m6-standup` | 15000，已导出 5500 与 7500 | `stand` | 见下 |
| 前滚翻。按键技能，不是持续步态 | 笔记未写机器 | `Mjlab-Roulade-Flat-MicroDuck` | `m6-roulade` | 配置默认 10000 | `roulade` | 日志文件名写成了 `VelStand-Flat_m6.log`，任务本身是翻滚 |

起立的两份导出：

| checkpoint | ONNX |
|---|---|
| `logs/rsl_rl/microduck_stand/2026-09-27_08-38-07_m6-standup/model_5500.pt` | `StandUp-Flat_4090D_5500s.onnx` |
| `logs/rsl_rl/microduck_stand/2026-09-27_08-37-12_m6-standup/model_7500.pt` | `standup.onnx` |

平地 VelStand 笔记里先有一次不带热启动的启动，随后用 `MICRODUCK_WARM_START=1` 从 `logs/rsl_rl/velstand/walk_expert/tfrere_base_walk.pt` 再启动。以后者为准：它保留走路权重，并把课程从 0 重新计。

三类槽位：

- **`walk`**：行走，兼跌倒后站起。平地与崎岖两份 VelStand 都占这一槽，同一时间只能留一个。
- **`stand`**：只负责起立并保持站立，不走路。
- **`roulade`**：X 键前滚翻，时间窗结束就交还步态。

## robotd.toml 怎么填

文件在机器人上的 `/etc/robot/robotd.toml`，改的是 `[policy]`。每个槽只放一个 onnx。改完 `systemctl restart robotd`，或用 `robotctl policy load <槽> <路径>` 热切换。`robotctl policy reset <槽>` 去掉覆盖，回到发行版默认。

### 推荐：用 VelStand 自己处理跌倒

上游默认就是这样：`walk` 指向 `velstand.onnx`，`stand = "none"`，`limp_fall` 保持 false。松开摇杆时仍走 walk 网络，零速度指令下它站着；摔倒时也是这一张网在恢复，避免卸力再交给另一张起立网时的抽搐。

平地：

```toml
[policy]
mode = "walk"
walk = "/home/radxa/policies/m6-VelStand-Flat.onnx"
stand = "none"
roulade = "/home/radxa/policies/m6-roulade.onnx"
```

崎岖路面只把 `walk` 换成 `m6-VelStand-Rough` 导出的 onnx。`stand` 仍是 `"none"`，`limp_fall` 不写（默认 false）。路径按板上实际文件名改。

### 备选：零指令时改用专用起立网

摇杆有速度时用 `walk`（VelStand，会走也会自己起来）。摇杆回到零时，若 `stand` 指向 StandUp，守护进程改跑起立网。这张网不会走路。只有在明确要「先卸力、停稳、再交给起立网」时，才把 `limp_fall` 设为 true，并把 `stand` 指到 `StandUp-Flat_4090D_5500s.onnx` 或 `standup.onnx`。

```toml
[policy]
mode = "walk"
walk = "/home/radxa/policies/m6-VelStand-Flat.onnx"
stand = "/home/radxa/policies/StandUp-Flat_4090D_5500s.onnx"
roulade = "/home/radxa/policies/m6-roulade.onnx"
limp_fall = true
```

## 不要这样配

| 写法 | 结果 |
|---|---|
| 把 StandUp 写进 `walk` | 机器人失去行走，只剩起立和站着 |
| 把 VelStand 写进 `stand` | 零指令槽被一张带速度指令的走路网占用 |
| `walk` 同时写平地和崎岖 | 一个键只能有一个路径，后写的覆盖先写的 |
| 翻滚写进 `walk` 或 `stand` | 前滚翻只在按下 X 的时间窗里运行，窗口键是 `roulade` |

翻滚训练回合长 5.0 秒（含落地站稳）。守护进程默认只把翻滚网打开 1.0 秒（`roulade_duration`）。真机上一次滚不完时再加大这个数。

本仓库这批策略训练时没有动作低通。上机时 `action_scale = 1.0`，`head_lowpass` 与 `legs_lowpass` 用 `1.0`，与这次训练一致。上游注释里的 0.9 / 0.5 / 0.7 是官方 alpha 策略的值，不要套到这四份 onnx 上。
