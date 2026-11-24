# XL330 的 KP=200 与 HD-1910 的 KP≈6：量纲为什么差这么多

> 2026-10-08。只回答一件事：这两个数字为什么不能直接比。
> 同一套 HD-1910 控制律里把 `kp_fw` 从 32 改成 200（硬 6.25 倍、饱和角从 11° 收到 1.8°）是另一件事，见 `docs/刚度混淆与解决方案_2026-09-17.md`。

200 和 6 都是写进固件的无量纲整数。寄存器宽度、出厂默认值、内部除数都不同。换成同一物理量（每弧度占空比，或堵转刚度 N·m/rad）之后，XL330 在表值 200 时比 HD-1910 寄存器 32 更软。

## 1. 两边寄存器

### XL330（ROBOTIS X 系列）

Position P Gain，控制表地址 84，**2 字节，0–16383，出厂 400**。

[XL330-M288 e-Manual](https://docs.robotis.com/docs/dxl/model_reference/x_series/xl_series/xl330-m288/) 给出内部增益：

| 控制表 | 内部增益 | 换算 |
|---|---|---|
| Position D Gain (80) | \(K_{PD}\) | 表值 / 16 |
| Position I Gain (82) | \(K_{PI}\) | 表值 / 65536 |
| Position P Gain (84) | \(K_{PP}\) | **表值 / 128** |

表值 200 → 内部 \(K_{PP} = 1.5625\)；出厂 400 → \(3.125\)。这个内部增益已经把采样时间折进去了，仍然不是 N·m/rad。PWM Limit（地址 36）出厂 **885**，单位 0.113%，885 = 100% 占空比。

### HD-1910 / STS（飞特）

位置环 P 系数，地址 21（RAM 镜像 50），**1 字节，0–254**。本仓库真机回读（固件 3.46，mode 4）是 **Kp=32 / Kd=40 / Ki=0**，见 `docs/feetech_hd1910_servo_notes.md`。STS3215 社区内存表的出厂 P 也是 32（[REGISTER_REFERENCE.md](https://github.com/commanderfun/STS3215/blob/main/REGISTER_REFERENCE.md)）。

飞特规格书只写“PID，可改”，不公布 ROBOTIS 那种除数。社区反推互相不一致：

- [aifitlab 内存表](https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis) 写 P 再除以 4、D 再除以 8。
- 本仓库 HLS-2909 探针是 P 缩放 1/8、D 缩放 1/4，`error_gain=0.166`（`vendor/bam/bam/feetech/actuator.py` 的 `HLS2909Actuator`）。
- HD-1910 **mode 4** 台架回归两种都不成立：36 条未饱和样本给出 `duty = 5.2 · Δq`（寄存器仍是 32，R²≥0.94），所以 `error_gain = 5.2/32 = 0.163`，记成“有效 P ≈ kp/6.1”。见 `docs/feetech_hd1910_pendulum_results.md`。

仓库里看到的“6 左右”就是这个有效增益（32/6.1 ≈ 5.2），或下面第 3 节那份拟合里的 `kp_fw=5`。寄存器本身仍是 32。

## 2. 换成同一物理量

堵转时 \(\tau = k_t \cdot V_{in} \cdot \mathrm{duty} / R\)，刚度 \(K = (\mathrm{d\,duty}/\mathrm{d}\Delta q) \cdot V_{in} \cdot k_t / R\)。

### XL330 的 BAM 模型

`vendor/bam/bam/dynamixel/actuator.py` 用的除数是实测 256，手册写的是 128：

```python
XL330_KP_DIVISOR = 256  # Empirically observed for XL330 (manual mentions 128)
XL330_PWM_LIMIT = 885
error_gain = (4096 / (2π)) / (256 * 885)  # ≈ 0.002877
# duty = Δq * kp * error_gain
```

`xl330/m6.json`：\(k_t = 0.346\,\mathrm{N·m/A}\)，\(R = 2.502\,\Omega\)。

| 设定 | duty / rad | 堵转刚度 |
|---|---:|---:|
| 表值 200，5.0 V | 0.575 | **0.40 N·m/rad** |
| 表值 200，7.4 V | 0.575 | 0.59 N·m/rad |
| 出厂 400，5.0 V | 1.151 | 0.80 N·m/rad |
| 若改用手册 /128（表值 200，5.0 V） | 1.151 | 0.80 N·m/rad |

### HD-1910 的台架控制律

`duty = Δq · kp · error_gain`，`kp=32`，`error_gain=0.163`。`hd1910/m5.json` 钉死 \(k_t=0.692\)、\(R=3.75\)。

| 设定 | duty / rad | 堵转刚度 |
|---|---:|---:|
| 寄存器 32，7.5 V | 5.22 | **7.2 N·m/rad** |
| 寄存器 32，5.0 V | 5.22 | 4.8 N·m/rad |

按手册除数把 XL330 再乘 2，它在表值 200 时仍然比 HD-1910 寄存器 32 软大约一个数量级。HD-1910 的力矩常数更大（0.69 vs 0.35 N·m/A），同样占空比给出的力矩也更大；两边的差距主要来自 `error_gain`（0.163 vs 0.00288，约 57 倍），寄存器整数 200 vs 32 只贡献约 6 倍，而且方向相反，抵消不掉。

把 XL330 的 200 **原样写进 HD-1910 这条律**（`error_gain=0.163` 不动）会比真机寄存器 32 硬 **6.25 倍**。`docs/training_config_velocity_hd1910.md` 里“不是 XL330 的 200（会硬 6 倍）”指的是这件事。

## 3. 本仓库现在写成 5 的原因

`src/mjlab_microduck/robot/microduck_constants.py` 的默认是 `KP_FW = 5`，注释写明：这是 xgoduck 那份拟合使用的 P，跟 reg50=32、跟 XL330 的 200 都不是同一个数。

那份 `hd1910/m6.json` 的 `actuator` 字段是 `sts3215`，控制律是

```text
duty = Δq · kp_fw · 0.166 · error_gain_ratio
```

JSON 里 `error_gain_ratio = 1.322`，\(k_t=0.624\)，\(R=4.91\)。`kp_fw=5`、7.5 V 时：

- duty / rad = \(5 \times 0.166 \times 1.322 = 1.10\)
- 堵转刚度 ≈ **1.0 N·m/rad**

这靠近 XL330 的 0.4–0.6，离第 2 节台架实测的 7.2 还差一截。5 是“STS3215 律下拟合吸收出来的 kp”，不是把飞特寄存器从 32 改成了 5。

## 4. 别人怎么说

有公开来源专门讲“这个整数不能跨固件抄”：

- [Rhoban BAM 采集文档](https://bam.readthedocs.io/en/latest/identification/acquisition.html)：一种固件上合理的 P，换一种固件可能高到或低到不能用。做法是取**该厂家自己的出厂 kp**，扫 `[kp/6, kp/4, kp/2, kp]`。XL330 出厂 400，他们的 `all_record` 扫的就是 `[50, 100, 200, 300, 400]`。官方行走用的 200 就是这组里的 `kp/2`。
- 同一套 BAM 源码注明 XL330 除数“手册写 128，实测是 256”（本仓库 `dynamixel/actuator.py` 保留了这句）。
- [舵机 PID 说明（aitoy-news）](https://aitoy-news.com/inside-servo-pid-tuning-for-humanoid-toy-robots) 把 XL330 的 /128 和飞特放在一起：写进寄存器的数不是控制器真正用的增益；同一数字在不同系列、不同固件上含义不同。

能横向比较的量是 duty/rad 或 N·m/rad。按这个口径：XL330 表值 200 ≈ 0.4 N·m/rad（5 V），HD-1910 寄存器 32 ≈ 7 N·m/rad（7.5 V）；仓库里的“6 左右”是飞特有效增益 kp/6.1，或 STS 拟合里的 `kp_fw=5`。
