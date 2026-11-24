"""Check the installed OpenMD training assets or smoke-test one exported actor."""

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
M6_SHA256_LF = "aaa3cd7403914d88b23c0e215b9eadcd3d2781117f877ce5861d7532799af322"
TASK_SPECS = {
    "Mjlab-StandUp-Flat-OpenMD": "get_openmd_stand_spec",
    "Mjlab-Velocity-Flat-OpenMD": "get_openmd_walk_spec",
    "Mjlab-Velocity-Flat-Backlash-OpenMD": "get_openmd_walk_backlash_spec",
    "Mjlab-VelStand-Flat-OpenMD": "get_openmd_allcollisions_spec",
    "Mjlab-SitStand-Flat-OpenMD": "get_openmd_stand_spec",
    "Mjlab-BallKick-Flat-OpenMD-Right": "get_openmd_stand_spec",
    "Mjlab-Roulade-Flat-OpenMD": "get_openmd_stand_spec",
}


def check_training():
    import bam
    from mjlab.tasks.registry import list_tasks, load_env_cfg
    from mjlab_microduck.robot import microduck_constants

    for module, directory in ((bam, ROOT / "vendor/bam"), (microduck_constants, ROOT / "src")):
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(directory):
            raise ValueError(f"{module.__name__} loaded from {origin}, expected {directory}")
    missing = TASK_SPECS.keys() - set(list_tasks())
    if missing:
        raise ValueError(f"Missing OpenMD tasks: {', '.join(sorted(missing))}")

    params = microduck_constants._BAM_ACTUATOR_KWARGS
    if (params["motor_name"], params["model"], params["kp_fw"]) != ("hd1910", "m6", 5):
        raise ValueError("Expected HD-1910 m6, kp_fw=5")
    path = Path(bam.__file__).parent / "params/hd1910/m6.json"
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != M6_SHA256_LF:
        raise ValueError(f"Unexpected m6 calibration: {path} ({digest})")
    print("HD-1910 m6 OK", path)

    compiled = set()
    for task, spec_name in TASK_SPECS.items():
        factory = load_env_cfg(task).scene.entities["robot"].spec_fn
        if factory.__name__ != spec_name:
            raise ValueError(f"{task}: expected {spec_name}, got {factory.__name__}")
        if spec_name not in compiled:
            factory().compile()  # Resolves the MJCF and loads the referenced meshes.
            compiled.add(spec_name)
        print(task, spec_name)


def check_onnx(path):
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 61]:
        raise ValueError("Expected one [1, 61] observation input")
    if len(outputs) != 1 or outputs[0].shape != [1, 14]:
        raise ValueError("Expected one [1, 14] action output")
    rng = np.random.default_rng(42)
    for obs in (np.zeros((1, 61), dtype=np.float32), rng.normal(0, 0.1, (1, 61)).astype(np.float32)):
        actions = session.run(None, {inputs[0].name: obs})[0]
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite policy output")
    print(f"ONNX [1,61] -> [1,14] OK   SHA-256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, help="Check a single exported ONNX on CPU")
    args = parser.parse_args()
    if args.onnx is None:
        check_training()
    else:
        check_onnx(args.onnx)
