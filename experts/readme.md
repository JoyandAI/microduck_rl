# tfrere 参考策略

这 9 个文件是 [tfrere](https://huggingface.co/tfrere) 发布的 Microduck 策略 checkpoint（各仓库的 `base/model.pt`）。`.pt` 被仓库 `.gitignore` 忽略，克隆之后要按下面的地址自己下载。

VelStand-Flat 只用前两行。`tfrere_base_stand.pt` 是倒下时的老师（倾斜 > 35°，`distill.py` 的 `checkpoint_path`）。`tfrere_base_walk.pt` 是站直时的老师（倾斜 < 25°，`anchor_checkpoint_path`），同时也是热启动权重。两份不要对调。其余文件这个任务不加载。翻滚任务从零训练，不读 `tfrere_base_roulade.pt`。操作步骤见 `docs/autodl_standup_retrain_guide.md` §3.7。

| 文件 | 来源 |
| --- | --- |
| `tfrere_base_stand.pt` | https://huggingface.co/tfrere/microduck-academy-base-stand/resolve/main/base/model.pt |
| `tfrere_base_walk.pt` | https://huggingface.co/tfrere/microduck-move-base-walk/resolve/main/base/model.pt |
| `tfrere_sitstand.pt` | https://huggingface.co/tfrere/microduck-move-base-sitstand/resolve/main/base/model.pt |
| `tfrere_play_dead_flat.pt` | https://huggingface.co/tfrere/microduck-move-play-dead-flat/resolve/main/base/model.pt |
| `tfrere_pecking_seeds.pt` | https://huggingface.co/tfrere/microduck-move-hen-pecking-seeds/resolve/main/base/model.pt |
| `tfrere_ground_pick.pt` | https://huggingface.co/tfrere/microduck-move-base-ground-pick/resolve/main/base/model.pt |
| `tfrere_kick_left.pt` | https://huggingface.co/tfrere/microduck-move-base-kick-left/resolve/main/base/model.pt |
| `tfrere_kick_right.pt` | https://huggingface.co/tfrere/microduck-move-base-kick-right/resolve/main/base/model.pt |
| `tfrere_base_roulade.pt` | https://huggingface.co/tfrere/microduck-move-base-roulade/resolve/main/base/model.pt |
