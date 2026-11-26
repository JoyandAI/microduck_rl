import csv
import importlib.util
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from mjlab_microduck import standup_replay as replay


@pytest.fixture(scope="module")
def ip():
    path = Path(__file__).resolve().parents[1] / "scripts/infer_policy.py"
    spec = importlib.util.spec_from_file_location("infer_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_actor(metadata):
    weight = helper.make_tensor("weights", TensorProto.FLOAT, [61, 14], np.eye(61, 14).flatten())
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weights"], ["action"])], "actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 14])], [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def load_actor(metadata):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    return ort.InferenceSession(make_actor(metadata), opts, providers=["CPUExecutionProvider"])


def test_task_selection():
    assert replay.select_task(None, replay.TASK_ID, True)
    assert replay.select_task(replay.TASK_ID, None, True)
    assert not replay.select_task(None, None, True)
    assert not replay.select_task(None, "Mjlab-VelStand-Flat-OpenMD", True)
    with pytest.raises(ValueError, match="task_id"):
        replay.select_task(replay.TASK_ID, "Mjlab-StandUp-Flat-MicroDuck", True)
    with pytest.raises(ValueError, match="standing"):
        replay.select_task(replay.TASK_ID, None, False)
    assert not replay.select_task(None, replay.TASK_ID, False)


def test_actor_and_targets_follow_the_registered_standup(ip):
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(replay.TASK_ID, play=True)
    model = mujoco.MjModel.from_xml_path(str(replay.SCENE))
    data = mujoco.MjData(model)
    session = load_actor({"task_id": replay.TASK_ID})
    policy = ip.PolicyInference(model, data, standing_session=session, standup_task=True,
                                delay_min_lag=1, delay_max_lag=1)
    assert policy.action_scale == cfg.actions["joint_pos"].scale
    assert cfg.observations["actor"].terms["joint_vel"].delay_min_lag == 1
    assert cfg.observations["actor"].terms["joint_vel"].delay_max_lag == 1
    assert tuple(cfg.observations["actor"].terms) == replay.OBSERVATION_NAMES
    robot = cfg.scene.entities["robot"]
    for joint, actual in zip(model.actuator_trnid[:, 0], policy.default_pose, strict=True):
        expected, = [value for expr, value in robot.init_state.joint_pos.items()
                     if re.fullmatch(expr, model.joint(joint).name)]
        assert actual == pytest.approx(expected)
    for pose in ("stand", "sit", "prone", "supine"):
        replay.set_start_pose(model, data, pose)
        obs = policy.get_observations()
        assert obs.shape == (61,)
        assert np.isfinite(obs).all()
        if pose == "stand":
            np.testing.assert_allclose(obs[6:20], 0, atol=1e-7)
        if pose in ("prone", "supine"):
            assert obs[3] == pytest.approx(1 if pose == "prone" else -1)
    data.qvel[policy.joint_qvel_indices] = .25
    mujoco.mj_forward(model, data)
    policy.infer()
    data.qvel[policy.joint_qvel_indices] = .5
    np.testing.assert_allclose(policy.get_observations()[20:34], .25)
    policy.apply_action(np.full(14, .25, dtype=np.float32))
    np.testing.assert_allclose(data.ctrl, policy.default_pose + .25 * policy.action_scale)
    policy.apply_action(np.full(14, .4, dtype=np.float32))
    np.testing.assert_allclose(data.ctrl, policy.default_pose + .25 * policy.action_scale)
    policy.apply_action(np.full(14, .1, dtype=np.float32))
    np.testing.assert_allclose(data.ctrl, policy.default_pose + .4 * policy.action_scale)


def test_exported_scale_and_conflicting_metadata():
    model = mujoco.MjModel.from_xml_path(str(replay.SCENE))
    session = load_actor({"action_scale": ",".join(["0.7"] * 14)})
    _, scale = replay.policy_settings(model, session, None)
    assert scale == pytest.approx(.7)
    with pytest.raises(ValueError, match="action-scale"):
        replay.policy_settings(model, session, 1.)
    for key in ("joint_names", "default_joint_pos", "observation_names"):
        with pytest.raises(ValueError, match=key):
            replay.policy_settings(model, load_actor({key: "9"}), None)


def test_tagged_cli_drives_standup_without_scene_or_layout_flags(ip, tmp_path, monkeypatch, capsys):
    path = tmp_path / "actor.onnx"
    path.write_bytes(make_actor({"task_id": replay.TASK_ID}))
    log = tmp_path / "observations.csv"
    monkeypatch.setattr(sys, "argv", ["infer_policy", "--standing", str(path), "--start-pose", "prone",
                                      "--save-csv", str(log)])
    monkeypatch.setattr(ip, "TerminalInput", lambda: nullcontext(SimpleNamespace(get_keys=lambda: [])))
    steps = iter([True, True, False])
    states = []

    def launch(model, data, **kwargs):
        def capture():
            states.append(data.qpos.copy())
        return nullcontext(SimpleNamespace(sync=capture, is_running=steps.__next__))

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    ip.main()
    assert "scene_stand.xml" in capsys.readouterr().out
    assert len(states) == 3 and np.isfinite(states).all()
    assert states[0][5] == pytest.approx(np.sqrt(.5))
    assert not np.array_equal(states[0], states[-1])
    with log.open() as stream:
        first, second = csv.DictReader(stream)
    for i in range(14):
        assert float(first[f"obs_{34+i}"]) == 0
        assert float(second[f"obs_{34+i}"]) == float(first[f"action_{i}"])
