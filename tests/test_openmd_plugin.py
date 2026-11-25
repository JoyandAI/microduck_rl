"""Exercise the installed task plugin and the call signatures used by its managers."""

import inspect
from importlib.metadata import entry_points

from mjlab.tasks.registry import list_tasks, load_env_cfg


def test_openmd_plugin_registers_executable_manager_terms():
    (plugin,) = entry_points(group="mjlab.tasks", name="mjlab_microduck")
    plugin.load()
    tasks = [name for name in list_tasks() if name.endswith("-OpenMD") or "-OpenMD-" in name]
    assert {"Mjlab-VelStand-Flat-OpenMD", "Mjlab-SitStand-Flat-OpenMD"} <= set(tasks)
    for task in tasks:
        for play in (False, True):
            cfg = load_env_cfg(task, play=play)
            for group in (cfg.rewards, cfg.events, cfg.terminations, cfg.curriculum):
                for name, term in group.items():
                    if term is None or not term.func.__module__.startswith("mjlab_microduck"):
                        continue
                    fn = term.func.__call__ if inspect.isclass(term.func) else term.func
                    try:
                        inspect.signature(fn).bind_partial(**term.params)
                    except TypeError as error:
                        raise AssertionError(f"{task} {name}: {error}") from error


def test_openmd_sitstand_replaces_a_microduck_head_contact():
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import configure_openmd_standing_support

    cfg = load_env_cfg("Mjlab-SitStand-Flat-OpenMD")
    (head,) = [sensor for sensor in cfg.scene.sensors if sensor.name == "head_ground_contact"]
    head.primary.pattern = "jaw_soft"
    configure_openmd_standing_support(cfg)

    (head,) = [sensor for sensor in cfg.scene.sensors if sensor.name == "head_ground_contact"]
    assert head.primary.pattern == "bottom_head_shell"
    assert cfg.rewards["head_ground"].params["sensor_name"] == head.name
