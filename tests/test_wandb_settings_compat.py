"""rsl_rl 5.0.1 still builds ``wandb.Settings(start_method="thread")``.

wandb 0.29.0 deleted that field and rejects unknown settings, so training died
in ``WandbSummaryWriter`` before the first iteration. ``train_hook`` drops the
argument at plugin import, which is how AutoDL runs wandb 0.30.0.
"""

import pytest

wandb = pytest.importorskip("wandb")

from mjlab_microduck.train_hook import ignore_removed_wandb_start_method


def test_settings_accepts_the_start_method_rsl_rl_still_passes() -> None:
    ignore_removed_wandb_start_method()
    # Second call must not wrap __init__ again.
    ignore_removed_wandb_start_method()
    wandb.Settings(start_method="thread")
