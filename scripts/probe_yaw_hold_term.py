#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接验证 `mdp.yaw_hold_at_zero_command` 的数学与门控（不依赖任何评测脚本的冻结约定）。

做法：建 play 环境 → 把 twist 指令范围全设成 (0,0)（采样即零）→ 用固定动作(HOME)站 2 s
→ 逐步调用该奖励项并打印分解量（standing 掩码 / 偏航误差 / 偏航角速度 / 返回值）；
再把指令设成 (0.2, 0, 0) 走 1 s，确认该项**恒为 0**（对行走零影响）。

用法：
    .venv/bin/python scripts/probe_yaw_hold_term.py
"""

from __future__ import annotations

import torch


def main() -> int:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks import mdp
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck", play=True)
    cfg.scene.num_envs = 4
    cfg.events.pop("push_robot", None)
    # 指令恒为零（不靠"冻结"这个约定）
    cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands["twist"].rel_standing_envs = 0.0
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    # 裸 ManagerBasedRlEnv 没有 RL 包装器注入的 extras["log"]，而部分奖励项会往里写
    # 训练指标（例：angular_momentum_penalty）⇒ 手动补一个空字典。
    unwrapped.extras.setdefault("log", {})
    asset_cfg = SceneEntityCfg("robot")

    def probe(tag: str, steps: int) -> None:
        acts = torch.zeros((unwrapped.num_envs, 14), device=unwrapped.device)
        vals, errs, rates, stands = [], [], [], []
        for _ in range(steps):
            unwrapped.step(acts)
            v = mdp.yaw_hold_at_zero_command(unwrapped, asset_cfg)
            vals.append(v.mean().item())
            cmd = unwrapped.command_manager.get_command("twist")
            stands.append(bool((cmd.abs() <= 0.01).all().item()))
            err = unwrapped.scene["robot"].data.root_link_quat_w
            rates.append(float(unwrapped.scene["robot"].data.root_link_ang_vel_b[:, 2].abs().mean()))
            errs.append(float(err[:, 0].abs().mean()))
        print(f"  [{tag}] 指令 {cmd[0].tolist()} standing={stands[-1]} "
              f"项值 mean={sum(vals)/len(vals):+.4f} (首 {vals[0]:+.4f} → 末 {vals[-1]:+.4f}) "
              f"|wz|均值={sum(rates)/len(rates):.3f} rad/s")

    print("=== 零指令站立（该项应≈+1 − |wz|）===")
    probe("stand", 100)

    print("\n=== 走 0.2 m/s（该项应恒为 0）===")
    cfg.commands["twist"].ranges.lin_vel_x = (0.2, 0.2)
    unwrapped.command_manager.get_term("twist").command[:, 0] = 0.2
    probe("walk", 50)

    print("\n[判据] 站立非零且>0.5 ⇒ 项生效；走路全 0 ⇒ 门控正确（对步态零影响）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
