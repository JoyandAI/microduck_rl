# Sitstand policy — how the current one was trained

Commanded sit ↔ stand in one policy (posture flag in the twist vx slot: 0 = stand, 1 = sit),
head commandable in both postures. Task `Mjlab-SitStand-Flat-MicroDuck`, cfg
`src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py` (the module docstring holds the
reward design and its lesson arc).

**Current policy: wandb `pollen-robotics/mjlab_microduck/vfhxgrum` @ `model_2250.pt`**
(validated on the real robot 2026-10-01). It is a two-stage lineage — base run, then a
same-task **resume** with the seated fall-back hardening.

## Stage 1 — base run `jngaedcs` (from scratch)

| | |
|---|---|
| code | commit `c832147` ("better sit stand", 2026-08-12) |
| host | COACH, RTX 4090 |
| checkpoint used | `model_1250.pt` — the previously deployed `alpha_sitstand.onnx` (Hub `pollen-robotics/microduck-policies` and `microduck/policies/`, identical md5 `ae90646…`, exact weight match) |

```bash
uv run train Mjlab-SitStand-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 10000
```

Gentle sit and rise, seated head control. **Known defect:** an abrupt head move while seated
tips the robot backward; it stays leaned ~30° in the sit pose and the neck servo overloads
holding the head.

## Stage 2 — resume `vfhxgrum` (seated fall-back hardening)

| | |
|---|---|
| code | commit `77ffadf` ("fix sitstand fall back"), branch `sitstand_no_fall_back` |
| host | COACH2, RTX 5090 |
| start | `jngaedcs` @ `model_1250.pt`, resumed (iteration counter 1250 kept) |
| checkpoint used | **`model_2250.pt`** (run killed at 2761; 2250 was good on the robot) |

```bash
uv run train Mjlab-SitStand-Flat-MicroDuck --env.scene.num-envs 4096 \
  --wandb-run-path pollen-robotics/mjlab_microduck/jngaedcs --wandb-checkpoint-name model_1250.pt \
  --agent.resume True --agent.max_iterations 2250
```

### Why a resume (and not `MICRODUCK_WARM_START=1`)

It is the **same task**, so restoring `common_step_counter` is what we want: every existing
curriculum (head range, DR, pushes, action rate, rise cap) continues exactly where the base
run left it at 1250, and the new perturbations are scheduled on the absolute step counter so
they start 250 iterations after the resume. `--agent.max_iterations` counts iterations
**added** to the loaded one (2250 → would end at 3500). `MICRODUCK_WARM_START=1` would
instead restart every curriculum at stage 0 under a policy trained past it — only for
cross-task seeding.

### What changed (all in commit `77ffadf`)

Root cause, measured in sim before changing anything (`claude_experiments/sitstand_head_bench.py`,
`sitstand_tip_probe.py`): the training head command only resamples every few seconds (never
fast), and no leaned-back seated state ever occurs, so there was neither the stimulus nor any
recovery data.

| term | kind | what | schedule (iter) |
|---|---|---|---|
| `seated_head_jerk` (`seated_head_command_jerk`) | interval event, 0.3–0.8 s | seated envs only: head command sign-flipped (fast wiggle) or re-sampled in the live range (jump) | prob 0 → **0.2 @1500** → **0.4 @2000** |
| `seated_tip` (`seated_backward_tip`) | interval event, 3–6 s, p=0.5 | seated envs only: backward pitch kick (−ω_y, ±30% roll mix) | 0 → (1.0, 2.5) rad/s @1500 → (1.5, 3.25) @2000 → (2.0, 4.0) @2500 |
| `neck_strain_leaned` (`neck_torque_when_tilted`) | cost, weight **−0.5** | mean (\|τ_neck\|/0.1)² × sit-blend × backward-lean × tilt ramp 12°→25° | always on (zero when upright/standing/rising/leaning forward) |

"Seated" = sit flag on and the posture slew complete (α ≥ 0.98), so neither event touches
standing or transitions. The chosen `model_2250` was trained with the 0.4 jerk stage and the
(1.5, 3.25) rad/s tip stage; the (2.0, 4.0) stage never reached it.

### What the training log looked like

Mean reward 140 → 125 at 1500 and 131 → 116 at 2000 (each stage makes sitting still harder),
recovering within each stage. `neck_strain_leaned` −0.11 right after 1500 → −0.013 by 1990
(the policy stopped straining). All penalties ≤ 0 throughout, no `nan_state` resets.

### Sim eval (`claude_experiments/sitstand_fallback_eval.py`, seated, 128 envs × 30 s, perturbations at final strength)

| | jngaedcs@1250 (old) | vfhxgrum@2000 |
|---|---|---|
| head jerks only — time leaned back > 20° | 21% | **3.1%** |
| head jerks only — envs leaned at the end | 32% | **3.1%** |
| + tip kicks — time leaned back | 47% | 21% |
| neck \|τ\| while leaned | 0.20 Nm | **0.045 Nm** |

Recovery from a hard tip kick was still ~0 at 2000. The real-robot head-move symptom was
gone at 2250 regardless.

## Reproduce / deploy

```bash
uv run scripts/export.py Mjlab-SitStand-Flat-MicroDuck \
  --wandb-run-path pollen-robotics/mjlab_microduck/vfhxgrum --checkpoint 2250 --onnx-file sitstand.onnx
uv run python claude_experiments/sitstand_fallback_eval.py sitstand.onnx [--no-tip]   # before/after battery
```

Always export through `scripts/export.py` (obs normalizer is baked into the ONNX).

## Open item

The gamepad (`padd --max-head`, default 2.5 rad) can command far beyond the trained head range
(±1.10 neck/head pitch, ±1.40 yaw, ±0.31 roll) and mostly beyond the joint limits. Clamp the
head command in `robotd` to the trained range — commands into a joint stop are themselves a
neck-overload source. Widening the trained range (yaw to ~±2.5, pitch-down to ~−1.7) is a
separate, later retrain.
