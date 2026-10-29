@echo off
cd /d "%~dp0"
python check_pipeline.py --project
if errorlevel 1 (
  echo.
  echo Project check failed. Complete calibration and install dependencies first.
  pause
  exit /b 1
)
python gripper_grasp_control.py
pause
