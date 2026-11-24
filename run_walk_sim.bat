@echo off
rem ==========================================================================
rem  OpenMD scene + VelStand ONNX in MuJoCo (one-click launcher)
rem  Repo : E:\dev\microduck_rl
rem  Venv : E:\dev\microduck_rl\.venv-infer  (python 3.12, mujoco 3.10.0,
rem         onnxruntime, better-actuator-models)
rem
rem  Scene is OpenMD scene_walk.xml (soles only). Policy is the m6 VelStand
rem  checkpoint. BAM stays hd1910/m6, kp_fw=5, vin=7.5 V — the training
rem  actuator. Add --no-bam to switch to the XML position actuators.
rem  Extra args are passed through, e.g.:  run_walk_sim.bat --no-bam --vin 4.8
rem
rem  IMPORTANT: run from a REAL console (double-click this .bat).
rem  Keyboard input does NOT work inside mintty / Git-Bash terminals.
rem  Keys: arrows = velocity commands, A/E turn, SPACE coast, P push, Q quit
rem ==========================================================================
setlocal
set PYTHON=E:\dev\microduck_rl\.venv-infer\Scripts\python.exe
set SCRIPT=E:\dev\microduck_rl\scripts\infer_policy.py
set ONNX=E:\SynologySync\02-personal\10-创业\99-卓研智趣\MicroDuck\训练模型\20260930_m6-VelStand-Flat.onnx
set SCENE=E:\dev\microduck_rl\src\mjlab_microduck\robot\openmd\scene_walk.xml

rem  --delay 1 2 : 1-2 control steps (20-40 ms) approximates training's
rem  4-7 SUBSTEP command delay (20-35 ms).
"%PYTHON%" "%SCRIPT%" --walking "%ONNX%" --new-cmd-obs --scene "%SCENE%" --delay 1 2 %*

endlocal
