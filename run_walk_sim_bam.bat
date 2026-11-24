@echo off
rem ==========================================================================
rem  Alias of run_walk_sim.bat. Since the 2026-09 develop branch, BAM
rem  actuation (hd1910/m5, kp_fw=32, vin=5.0 V) is the DEFAULT in
rem  infer_policy.py, so no extra flag is needed ¡ª this launcher exists to
rem  keep the old entry point working. For the idealized XML position
rem  actuators instead, run:  run_walk_sim_bam.bat --no-bam
rem ==========================================================================
call "%~dp0run_walk_sim.bat" %*
