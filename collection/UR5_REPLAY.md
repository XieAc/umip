# XAC episode replay on UR5

Deployment edition: use the complete `umip` directory. The replay now also
imports `xac_config.py` and `calibrate_gripper_width.py`. Every live command below
must additionally supply `--tcp-offset TX TY TZ RX RY RZ` using the measured
flange-to-gripper transform (metres/radians) configured on the pendant. See
`README.md` for the current complete commands. Copying only this script is no
longer sufficient.

Run `replay_xac_ur5.py` on the computer physically connected to the UR5 and,
when needed, the XC330 U2D2. The input is one `build_dataset.py` episode, not a
training Zarr archive.

## Install on the 3060 computer

Copy these files and directories while preserving their relative layout:

```text
replay_xac_ur5.py
xac_config.py
calibrate_gripper_width.py
calibration/
dataset/replay/001/data/*.npy
dataset/replay/001/metadata/*
```

All current calibration files are managed in `calibration/`; see
`calibration/README.md`. Default paths come from `xac_config.py` and do not
depend on the working directory. Set `XAC_CALIBRATION_ROOT` before starting
the program if the complete calibration directory has been relocated.

Activate the deployment Conda environment and install the hardware bindings:

```bash
conda activate env_wyd_umi
pip install numpy scipy ur-rtde dynamixel-sdk
```

Put the UR5 in Remote Control mode, release the emergency stop and confirm the
3060 computer can reach the robot:

```bash
ping ROBOT_IP
```

## Required sequence

First check the recording without contacting hardware. This reports motion
relative to the recorded start pose:

```bash
python replay_xac_ur5.py \
  --episode /home/eirl3060/xac_dataset/replay/001
```

Then connect read-only to the UR5 so the report includes the actual base-frame
workspace. Replace both Z limits with values measured for the current cell:

```bash
python replay_xac_ur5.py \
  --episode /home/eirl3060/xac_dataset/replay/001 \
  --robot-ip ROBOT_IP \
  --minimum-base-z-m SAFE_MIN_Z \
  --maximum-base-z-m SAFE_MAX_Z \
  --max-distance-from-start-m 0.05 \
  --report logs/001_dry_run.json
```

The first live test controls only the arm and stops after three seconds of wall
time. `speed-scale 0.1` stretches a 10 Hz source interval from 0.1 s to 1.0 s:

```bash
python replay_xac_ur5.py \
  --episode /home/eirl3060/xac_dataset/replay/001 \
  --robot-ip ROBOT_IP \
  --enable-motion \
  --speed-scale 0.1 \
  --max-duration-s 3 \
  --minimum-base-z-m SAFE_MIN_Z \
  --maximum-base-z-m SAFE_MAX_Z \
  --max-distance-from-start-m 0.05 \
  --trace logs/001_arm_3s.json
```

The program asks for the exact text `RUN` immediately before sending motion.
Press `Ctrl+C` to call `servoStop` and terminate the RTDE control script.

Only after arm-only direction and bounds are correct, add the gripper. The
gripper must already be within 80 ticks of the recorded first width; otherwise
the program aborts before enabling its torque:

```bash
python replay_xac_ur5.py \
  --episode /home/eirl3060/xac_dataset/replay/001 \
  --robot-ip ROBOT_IP \
  --enable-motion \
  --enable-gripper \
  --gripper-port /dev/ttyUSB0 \
  --gripper-baudrate 57600 \
  --gripper-calibration calibration/gripper_width.json \
  --speed-scale 0.1 \
  --max-duration-s 3 \
  --minimum-base-z-m SAFE_MIN_Z \
  --maximum-base-z-m SAFE_MAX_Z \
  --max-distance-from-start-m 0.05
```

Do not use the old `scripts_real/eval_replay_real_robot.py` for this dataset. It
assumes a network WSG gripper, a 59.94 Hz source and absolute replay-buffer
poses. This script anchors the recorded motion with:

```text
T_B_G(t) = T_B_G_start @ inverse(T_W_G_start) @ T_W_G(t)
```
