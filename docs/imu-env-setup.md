# IMU 环境安装(Pi 5 台架 + microduck,越简越好)

IMU: **LSM6DSV16X** @ `0x6B`,总线 `i2c-1`,Pi 3V3 供电。

## 1. Pi 侧

```bash
# 1) 启用 i2c(/boot/firmware/config.txt,无则加;改后重启)
dtparam=i2c_arm=on,i2c_arm_baudrate=50000

# 2) 工具
sudo apt-get update && sudo apt-get install -y i2c-tools
pip install smbus2        # 或 sudo apt-get install -y python3-smbus2

# 3) IMU 总线别名 /dev/i2c-imu -> i2c-1
sudo tee /etc/udev/rules.d/99-robot-i2c-imu.rules >/dev/null <<'EOF'
KERNEL=="i2c-1", SUBSYSTEM=="i2c-dev", SYMLINK+="i2c-imu"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger --sysname-match=i2c-1
```

自检:

```bash
i2cdetect -y 1                  # 0x6b 出现
python3 -c "import smbus2; print(hex(smbus2.SMBus(1).read_byte_data(0x6b,0x0f)))"   # 0x70
```

## 2. 构建机(交叉编译 aarch64,一次性)

```bash
rustup target add aarch64-unknown-linux-gnu
# zig 0.14.x x86_64-linux: https://ziglang.org/download/ 下载后解压到 ~/.local/zig/
cargo install cargo-zigbuild
export PATH="$HOME/.cargo/bin:$HOME/.local/zig/zig-x86_64-linux-0.14.1:$PATH"
```

## 3. 构建 → 部署 → 读取(每次)

```bash
cd <microduck 仓库>
cargo zigbuild --target aarch64-unknown-linux-gnu.2.31 -p duck-control --example imu_probe
scp target/aarch64-unknown-linux-gnu/debug/examples/imu_probe joyandai@<pi>:~/microduck/
ssh joyandai@<pi> '~/microduck/imu_probe /dev/i2c-imu 0x6B'
```

## 4. 说明

- 期望输出:`ready=true`、四元数实时、`0 errors`。
- `robotd.toml` 的 `[imu]`:`bus = "/dev/i2c-imu"`,`address = 0x6B`(本台架;量产板为 0x6A)。
- `duck-control/src/imui2c.rs` 需包含两处修复:
  1. `init()` 里 `fifo_gy_batch_set(FifoBatch::_120hz)`(否则 FIFO 无陀螺仪,永远组不出数据块);
  2. `I2cDev::transaction` 瞬时 NACK 重试 ×3(本机总线约 4–15% 随机丢 ACK)。
- 若仍有随机 NACK:检查接线/接触,并考虑切断 IMU 板载 I2C 上拉跳线(与 Pi 板载上拉并联过强)。
