#!/usr/bin/env python3
"""Build OpenMD MJCF variants from the exported openmicroduck.xml.

Reads config_mjcf_*.json in this directory. Joint names are renamed onto the
training layout (left leg, neck/head, right leg). STAND / SIT / FOLD angles
are copied from openmicroduck_flat_scene.xml unchanged.

    uv run python src/mjlab_microduck/robot/openmd/build_models.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ADD_BACKLASH = REPO / "src" / "mjlab_microduck" / "robot" / "microduck" / "add_backlash.py"

# Source joint name -> training name. Document order in openmicroduck.xml is
# already left leg, neck/head, right leg.
JOINT_RENAME = {
    "hip_yaw_left_joint": "left_hip_yaw",
    "hip_left_joint": "left_hip_roll",
    "upper_leg_left_joint": "left_hip_pitch",
    "leg_left_link_joint": "left_knee",
    "foot_left_joint": "left_ankle",
    "neck_link_joint": "neck_pitch",
    "head_pitch_link_joint": "head_pitch",
    "head_yaw_link_joint": "head_yaw",
    "head_roll_link_joint": "head_roll",
    "hip_yaw_right_joint": "right_hip_yaw",
    "hip_right_link_joint": "right_hip_roll",
    "upper_leg_right_joint": "right_hip_pitch",
    "leg_right_joint": "right_knee",
    "foot_right_joint": "right_ankle",
}

BODY_RENAME = {
    "base_link": "trunk_base",
    "neck_link": "neck",
    "head_pitch_link": "neck_pitch",
    "head_yaw_link": "yaw_roll_motion",
    "head_roll_link": "bottom_head_shell",
}

# Bodies the velocity / velstand recipes look up by exact name. bearing_roll is
# a historical mis-label of a hip link in the MicroDuck head-CoM list; a tiny
# body keeps that lookup from failing without moving any real mass.
TRAINING_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "bottom_head_shell", "bearing_roll")

# Floor-contact meshes for the stand-up model (soles are always kept).
STAND_MESHES = {
    "sole_left",
    "sole_right",
    "leg_left",
    "leg_right",
    "left_shell",
    "right_shell",
    "trunk_base",
    "18650_2s",
    "hip_left",
    "hip_right",
    "upper_leg_left",
    "upper_leg_right",
    "yaw2roll",
    "jaw",
    "bottom_head_shell",
    "top_head_shell",
    "foot_left",
    "foot_right",
    "ankle_left",
    "ankle_right",
    "neck",
    "neck_pitch",
}

SERVO_JOINTS = list(JOINT_RENAME.values())


def _configs() -> list[dict]:
    out = []
    for path in sorted(HERE.glob("config_mjcf_*.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg["_path"] = path
        out.append(cfg)
    return out


def _source_paths(cfg: dict) -> tuple[Path, Path]:
    xml = REPO / Path(cfg["source_xml"])
    scene = REPO / Path(cfg["source_scene"])
    if not xml.is_file() or not scene.is_file():
        raise SystemExit(
            f"OpenMD source MJCF not found.\n  xml:   {xml}\n  scene: {scene}\n"
            "These paths are recorded in config_mjcf_*.json."
        )
    return xml, scene


def _copy_meshes(src_xml: Path) -> None:
    src = src_xml.parent / "meshes" / "stl"
    dst = HERE / "meshes" / "stl"
    if not src.is_dir():
        raise SystemExit(f"OpenMD meshes not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and any(dst.glob("*.stl")):
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _parents(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in list(parent)}


def _is_collision(geom: ET.Element) -> bool:
    return geom.get("contype") != "0"


def _disable(geom: ET.Element) -> None:
    geom.set("contype", "0")
    geom.set("conaffinity", "0")


def _apply_collision(root: ET.Element, mode: str) -> None:
    parents = _parents(root)
    servo_counts: dict[str, int] = {}
    named: dict[str, int] = {}
    for geom in root.iter("geom"):
        if not _is_collision(geom):
            continue
        mesh = geom.get("mesh") or ""
        if mesh in ("sole_left", "sole_right"):
            geom.set(
                "name",
                "left_foot_collision" if mesh == "sole_left" else "right_foot_collision",
            )
            geom.set("class", "collision")
            if mode == "stand":
                geom.attrib.update(contype="1", conaffinity="1", condim="3", priority="1")
            elif mode == "all":
                geom.attrib.update(contype="1", conaffinity="0")
            continue
        if mode == "feet":
            _disable(geom)
            continue
        if mode == "stand" and mesh not in STAND_MESHES:
            _disable(geom)
            continue
        if mode == "all" and mesh.startswith("hd1910"):
            body = parents[geom].get("name") or "body"
            k = servo_counts.get(body, 0)
            servo_counts[body] = k + 1
            geom.set("name", f"{body}_{k}_servo_collision")
        elif mode in ("stand", "all") and mesh:
            # FULL_COLLISION selects `.*_collision`; the exporter leaves the
            # collision twin as `<mesh>.001`, which that regex does NOT match
            # (CollisionCfg then disabled every body geom, including the head).
            base = f"{mesh}_collision"
            n = named.get(base, 0)
            named[base] = n + 1
            geom.set("name", base if n == 0 else f"{base}_{n}")
        geom.set("contype", "1")
        geom.set("conaffinity", "1" if mode == "stand" else "0")
        if mode == "stand":
            geom.set("condim", "1")


def _add_sites(root: ET.Element) -> None:
    trunk = next(b for b in root.iter("body") if b.get("name") == "trunk_base")
    inertial = trunk.find("inertial")
    imu_pos = inertial.get("pos") if inertial is not None and inertial.get("pos") else "0 0 0"
    ET.SubElement(trunk, "site", {"name": "imu", "pos": imu_pos, "group": "3"})
    ET.SubElement(
        trunk,
        "body",
        {"name": "bearing_roll", "pos": "0 0 0"},
    ).append(
        ET.Element(
            "inertial",
            {
                "pos": "0 0 0",
                "mass": "0.0001",
                "diaginertia": "1e-8 1e-8 1e-8",
            },
        )
    )
    sole_pos = {}
    for geom in root.iter("geom"):
        if geom.get("name") in ("left_foot_collision", "right_foot_collision"):
            sole_pos[geom.get("name")] = geom.get("pos") or "0 0 0"
    foot_body = {
        "left_foot_collision": "foot_left_link",
        "right_foot_collision": "foot_right_link",
    }
    site_name = {
        "left_foot_collision": "left_foot",
        "right_foot_collision": "right_foot",
    }
    bodies = {b.get("name"): b for b in root.iter("body")}
    for geom_name, body_name in foot_body.items():
        ET.SubElement(
            bodies[body_name],
            "site",
            {"name": site_name[geom_name], "pos": sole_pos[geom_name], "group": "3"},
        )


def _insert_defaults(root: ET.Element) -> None:
    defaults = ET.Element("default")
    chosen = ET.SubElement(defaults, "default", {"class": "chosen_actuator"})
    ET.SubElement(
        chosen,
        "joint",
        {"damping": "0.053", "frictionloss": "0.0048", "armature": "0.0018"},
    )
    ET.SubElement(
        chosen,
        "position",
        {
            "kp": "0.55",
            "kv": "0.0",
            "forcerange": "-0.96 0.96",
            "ctrlrange": "-10.0 10.0",
        },
    )
    collision = ET.SubElement(defaults, "default", {"class": "collision"})
    ET.SubElement(
        collision,
        "geom",
        {"group": "3", "friction": "1 0.005 0.0001"},
    )
    root.insert(0, defaults)


def _add_actuators_and_sensors(root: ET.Element) -> None:
    actuators = ET.SubElement(root, "actuator")
    for name in SERVO_JOINTS:
        ET.SubElement(
            actuators,
            "position",
            {"class": "chosen_actuator", "name": name, "joint": name},
        )
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(
        sensors,
        "framequat",
        {"name": "orientation", "objtype": "site", "noise": "0.001", "objname": "imu"},
    )
    ET.SubElement(
        sensors,
        "gyro",
        {"name": "angular-velocity", "site": "imu", "noise": "0.005"},
    )
    ET.SubElement(sensors, "gyro", {"name": "imu_ang_vel", "site": "imu"})
    ET.SubElement(sensors, "velocimeter", {"name": "imu_lin_vel", "site": "imu"})
    ET.SubElement(sensors, "accelerometer", {"name": "imu_accel", "site": "imu"})
    ET.SubElement(sensors, "subtreeangmom", {"name": "root_angmom", "body": "trunk_base"})


def _rename(root: ET.Element) -> None:
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name == "base_link_free":
            joint.set("name", "trunk_base_freejoint")
            continue
        if name in JOINT_RENAME:
            joint.set("name", JOINT_RENAME[name])
            joint.set("class", "chosen_actuator")
    for body in root.iter("body"):
        name = body.get("name")
        if name in BODY_RENAME:
            body.set("name", BODY_RENAME[name])


def _build_robot(src_xml: Path, mode: str, out_path: Path) -> None:
    tree = ET.parse(src_xml)
    root = tree.getroot()
    root.set("model", "openmd")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler", {"angle": "radian"})
        root.insert(0, compiler)
    compiler.set("autolimits", "true")
    _rename(root)
    # Spawn height. The source body sits at the origin, so qpos=0 puts every
    # collision mesh through the floor and overflows nconmax before the first
    # reset. 0.12 m is the STAND keyframe height in openmicroduck_flat_scene.xml.
    trunk = next(b for b in root.iter("body") if b.get("name") == "trunk_base")
    trunk.set("pos", "0 0 0.12")
    _apply_collision(root, mode)
    _add_sites(root)
    _insert_defaults(root)
    _add_actuators_and_sensors(root)
    comment = ET.Comment(
        " Generated by build_models.py from openmicroduck.xml. "
        "Joint names follow the training layout. Do not hand-edit; re-run the builder. "
    )
    root.insert(0, comment)
    ET.indent(tree, space="  ")
    with out_path.open("w", encoding="utf-8", newline="\n") as stream:
        tree.write(stream, encoding="unicode", xml_declaration=True)


def _keyframe_numbers(scene_xml: Path) -> dict[str, list[float]]:
    root = ET.parse(scene_xml).getroot()
    out = {}
    for key in root.findall("./keyframe/key"):
        nums = [float(x) for x in key.get("qpos", "").split()]
        if len(nums) != 21:
            raise SystemExit(f"key {key.get('name')} has {len(nums)} qpos values, expected 21")
        out[key.get("name")] = nums
    for name in ("INIT", "STAND", "SIT", "FOLD"):
        if name not in out:
            raise SystemExit(f"source scene is missing keyframe {name}")
    return out


def _fmt(nums: list[float]) -> str:
    return " ".join(f"{v:.16g}" for v in nums)


def _write_scene(robot_file: str, keys: dict[str, list[float]], out_path: Path, backlash: bool) -> None:
    def qpos(name: str) -> str:
        free = keys[name][:7]
        joints = keys[name][7:]
        if not backlash:
            return _fmt(free + joints)
        mixed: list[float] = []
        for q in joints:
            mixed.extend((q, 0.0))
        return _fmt(free + mixed)

    def ctrl(name: str) -> str:
        return _fmt(keys[name][7:])

    lines = [
        '<mujoco model="scene">',
        f'  <include file="{robot_file}" />',
        "  <visual>",
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />',
        '    <rgba haze="0.15 0.25 0.35 1" />',
        '    <global azimuth="160" elevation="-20" />',
        "  </visual>",
        "  <asset>",
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />',
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />',
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />',
        "  </asset>",
        "  <worldbody>",
        '    <light pos="0 0 3.5" dir="0 0 -1" directional="true" />',
        '    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" />',
        "  </worldbody>",
        "  <!-- qpos angles copied from openmicroduck_flat_scene.xml.",
        "       Order: free xyz+quat, L hip_yaw/roll/pitch/knee/ankle,",
        "       neck_pitch/head_pitch/head_yaw/head_roll,",
        "       R hip_yaw/roll/pitch/knee/ankle. -->",
        "  <keyframe>",
    ]
    for name in ("INIT", "STAND", "SIT", "FOLD"):
        lines.append(f'    <key name="{name}" qpos="{qpos(name)}" ctrl="{ctrl(name)}" />')
    lines.append("  </keyframe>")
    lines.append("</mujoco>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_stand(scene_path: Path) -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    sole_z = []
    for name in ("left_foot_collision", "right_foot_collision"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        sole_z.append(float(data.geom_xpos[gid, 2]))
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    tz = float(data.xpos[trunk, 2])
    print(
        f"[openmd] {scene_path.name}: trunk z={tz:.4f} m, "
        f"sole origin z L/R={sole_z[0]:.4f}/{sole_z[1]:.4f} m"
    )


def main() -> int:
    cfgs = _configs()
    direct = [c for c in cfgs if "collision" in c]
    derived = [c for c in cfgs if "from_robot" in c]
    if not direct:
        raise SystemExit(f"no config_mjcf_*.json with a collision mode in {HERE}")
    src_xml, src_scene = _source_paths(direct[0])
    _copy_meshes(src_xml)
    keys = _keyframe_numbers(src_scene)
    for cfg in direct:
        out = HERE / f"{cfg['output_filename']}.xml"
        _build_robot(src_xml, cfg["collision"], out)
        scene = HERE / f"scene_{cfg['output_filename'].removeprefix('robot_')}.xml"
        _write_scene(out.name, keys, scene, backlash=False)
        print(f"[openmd] wrote {out.name} + {scene.name}")
        _check_stand(scene)
    for cfg in derived:
        src = HERE / cfg["from_robot"]
        out = HERE / f"{cfg['output_filename']}.xml"
        shutil.copyfile(src, out)
        proc = subprocess.run(
            [
                sys.executable,
                str(ADD_BACKLASH),
                str(out),
                "--backlash-deg",
                str(cfg.get("backlash_deg", 2.0)),
            ],
            check=False,
        )
        if proc.returncode != 0:
            return proc.returncode
        scene = HERE / f"scene_{cfg['output_filename'].removeprefix('robot_')}.xml"
        _write_scene(out.name, keys, scene, backlash=True)
        print(f"[openmd] wrote {out.name} + {scene.name}")
        _check_stand(scene)
    missing = [name for name in TRAINING_BODIES if name != "bearing_roll"]
    # bearing_roll is injected; the others must come from BODY_RENAME.
    if set(missing) - set(BODY_RENAME.values()):
        raise SystemExit(f"training body names not produced by BODY_RENAME: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
