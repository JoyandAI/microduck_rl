import subprocess
import sys
from pathlib import Path


CHECK = Path(__file__).resolve().parents[1] / "scripts/check_openmd_training.py"


def test_registered_walk():
    task = "Mjlab-Velocity-Flat-Backlash-OpenMD"
    result = subprocess.run(
        [sys.executable, str(CHECK), "--registered-task", task], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert task in result.stdout


def test_missing_task_fails():
    task = "Mjlab-Missing-OpenMD"
    result = subprocess.run(
        [sys.executable, str(CHECK), "--registered-task", task], capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert f"Missing registered tasks: {task}" in result.stderr


def test_preflight_propagates_plugin_import_error():
    # Exercise the installed entry point in a new interpreter, before mjlab discovers it.
    driver = '''
import runpy, sys
class BrokenConfig:
    def find_spec(self, fullname, path, target=None):
        if fullname == "mjlab_microduck.tasks.openmd_env_cfg":
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BrokenConfig())
runpy.run_path(sys.argv.pop(), run_name="__main__")
'''
    result = subprocess.run([sys.executable, "-c", driver, str(CHECK)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "ModuleNotFoundError: mjlab_microduck.tasks.openmd_env_cfg" in result.stderr
