"""OpenMD task and policy factories for the HD-1910 robot."""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg

from mjlab_microduck.robot.openmd_constants import (
    OPENMD_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    OPENMD_ALLCOLLISIONS_ROBOT_CFG,
    OPENMD_GROUND_PICK_ROBOT_CFG,
    OPENMD_STAND_ROBOT_CFG,
    OPENMD_WALK_BACKLASH_ROBOT_CFG,
    OPENMD_WALK_ROBOT_CFG,
)
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
    MicroduckBallKickLeftRlCfg,
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_env_cfg,
)
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    MicroduckGroundPickRlCfg,
    make_microduck_ground_pick_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    MicroduckSitStandRlCfg,
    configure_openmd_standing_support,
    make_microduck_sitstand_env_cfg,
)
from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    MicroduckStandUpRlCfg,
    make_microduck_standup_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    configure_openmd_locomotion,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.microduck_velstand_env_cfg import (
    MicroduckVelStandRlCfg,
    WARM_START,
    make_microduck_velstand_env_cfg,
)


def _use(cfg: ManagerBasedRlEnvCfg, robot) -> ManagerBasedRlEnvCfg:
    cfg.scene.entities["robot"] = robot
    return cfg


def make_openmd_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _use(make_microduck_standup_env_cfg(play=play), OPENMD_STAND_ROBOT_CFG)


def make_openmd_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play)
    configure_openmd_locomotion(cfg)
    return _use(cfg, OPENMD_WALK_ROBOT_CFG)


def make_openmd_walk_backlash_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return make_backlash_variant(
        make_openmd_velocity_env_cfg(play=play),
        OPENMD_WALK_BACKLASH_ROBOT_CFG,
    )


def make_openmd_velstand_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velstand_env_cfg(play=play, rough=rough)
    configure_openmd_locomotion(cfg, final_weights=WARM_START and not play)
    return _use(cfg, OPENMD_ALLCOLLISIONS_ROBOT_CFG)


def make_openmd_velstand_backlash_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    """VelStand on robot_allcollisions_backlash.xml.

    Same collision family as the plain VelStand tasks. The walk backlash model
    only has sole contacts, so a fall would have nothing to land on.
    """
    return make_backlash_variant(
        make_openmd_velstand_env_cfg(play=play, rough=rough),
        OPENMD_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    )


# World-up expressed in bottom_head_shell at the OpenMD STAND keyframe.
# MicroDuck stores the same measurement for its jaw_soft body.
_OPENMD_HEAD_TOP_AXIS = (0.0, 1.0, 0.0)


def _mark_openmd_roulade_head(env, env_ids) -> None:
    """Point the over-the-head latch at the OpenMD head body."""
    del env_ids
    env._roulade_head_body_name = "bottom_head_shell"
    env._roulade_head_top_axis = _OPENMD_HEAD_TOP_AXIS


def make_openmd_ball_kick_env_cfg(
    play: bool = False, kick_foot: str = "right"
) -> ManagerBasedRlEnvCfg:
    """One-foot kick on robot_stand.xml.

    The ball has to meet the leg, so this uses the stand collision set rather
    than the soles-only walk model. At STAND the right foot site is at
    y=-0.042 and the toe is near x=0.035; ``kick_foot="left"`` mirrors that
    offset to +y and plants the right foot as the support foot.
    """
    cfg = _use(
        make_microduck_ball_kick_env_cfg(play=play, kick_foot=kick_foot),
        OPENMD_STAND_ROBOT_CFG,
    )
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
    return cfg


def make_openmd_ground_pick_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    """Ground pick on robot_stand.xml.

    The MicroDuck payload hook reads body ``jaw_soft``. On this export that name
    is a geom; the beak hangs off ``bottom_head_shell``, which also carries the
    ``mouth_tip`` site added by ``get_openmd_ground_pick_spec``.
    """
    cfg = _use(
        make_microduck_ground_pick_env_cfg(play=play, rough=rough),
        OPENMD_GROUND_PICK_ROBOT_CFG,
    )
    cfg.rewards["mouth_payload_force"].params["asset_cfg"].body_names = (
        "bottom_head_shell",
    )
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
    return cfg


def make_openmd_sitstand_env_cfg(
    play: bool = False, rough: bool = False
) -> ManagerBasedRlEnvCfg:
    """Commanded sit and stand on robot_stand.xml.

    The head-load sensor reads contacts on ``bottom_head_shell`` while standing.
    """
    cfg = _use(
        make_microduck_sitstand_env_cfg(play=play, rough=rough),
        OPENMD_STAND_ROBOT_CFG,
    )
    configure_openmd_standing_support(cfg)
    return cfg


def make_openmd_roulade_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Roulade on robot_stand.xml.

    The MicroDuck latch reads body ``jaw_soft``. On this export that name is a
    geom; the head shells that touch the floor hang off ``bottom_head_shell``.
    """
    cfg = _use(make_microduck_roulade_env_cfg(play=play), OPENMD_STAND_ROBOT_CFG)
    for sensor in cfg.scene.sensors:
        if sensor.name == "head_ground_contact":
            sensor.primary.pattern = "bottom_head_shell"
    cfg.events["openmd_roulade_head"] = EventTermCfg(
        func=_mark_openmd_roulade_head,
        mode="startup",
    )
    # A roll plants many floor geoms at once. The velocity default (35) overflows.
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
    return cfg


OpenMDStandRlCfg = replace(
    MicroduckStandUpRlCfg,
    experiment_name="openmd_stand",
    run_name="openmd_stand",
)
OpenMDWalkRlCfg = replace(
    MicroduckRlCfg,
    actor=replace(
        MicroduckRlCfg.actor,
        distribution_cfg={**MicroduckRlCfg.actor.distribution_cfg, "init_std": 1.0},
    ),
    experiment_name="openmd_velocity",
    run_name="openmd_walk",
)
OpenMDVelStandRlCfg = replace(
    MicroduckVelStandRlCfg,
    experiment_name="openmd_velstand",
    run_name="openmd_velstand",
)
OpenMDRouladeRlCfg = replace(
    MicroduckRouladeRlCfg,
    experiment_name="openmd_roulade",
    run_name="openmd_roulade",
)
OpenMDBallKickRlCfg = replace(
    MicroduckBallKickRlCfg,
    experiment_name="openmd_ball_kick_right",
    run_name="openmd_ball_kick_right",
)
OpenMDBallKickLeftRlCfg = replace(
    MicroduckBallKickLeftRlCfg,
    experiment_name="openmd_ball_kick_left",
    run_name="openmd_ball_kick_left",
)
OpenMDSitStandRlCfg = replace(
    MicroduckSitStandRlCfg,
    experiment_name="openmd_sitstand",
    run_name="openmd_sitstand",
)
OpenMDGroundPickRlCfg = replace(
    MicroduckGroundPickRlCfg,
    experiment_name="openmd_ground_pick",
    run_name="openmd_ground_pick",
)
