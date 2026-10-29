# UR5 replay entry point

当前有效的重播实现集中在 [`replay/`](replay/)。完整的离线诊断、目标导出和逐步真机流程见 [`replay/README.md`](replay/README.md)。

- `replay/replay_xac_ur5.py`：离线检查、TCP 补偿和 UR5 轨迹重播；
- `replay/verify_tcp_transform.py`：TCP 方向、平移和旋转验证；
- `driver/xc330.py`：重播时使用的 XC330 接口；
- `driver/d435i.py`：可选的 D435i 启动接口。

日常操作使用根目录统一入口，不要运行 `replaydata1/` 中的旧版脚本：

```bash
python deploy.py replay --help
python deploy.py verify-tcp --help
```

先做离线轨迹诊断，不连接机器人：

```bash
python deploy.py replay \
  --episode replay/replaydata1/dataset/replay/001 \
  --diagnose \
  --dump-targets logs/replay/replay_001_targets.json \
  --speed-scale 0.1 \
  --max-distance-from-start-m 1.0
```

当前左臂真机示例：

```bash
python deploy.py replay \
  --episode replaydata1/dataset/replay/001 \
  --robot-ip 192.168.1.101 \
  --tcp-calibration calibration/tcp_gripper_01_gx_down_candidate.json \
  --minimum-base-z-m "$SAFE_MIN_Z" \
  --maximum-base-z-m "$SAFE_MAX_Z" \
  --max-distance-from-start-m 0.25 \
  --speed-scale 0.1 \
  --max-duration-s 90 \
  --enable-motion \
  --trace logs/replay/arm_only.json
```

需要同步 XC330 时增加 `--enable-gripper` 及其串口参数；需要启动相机时增加
`--enable-d435i`。完整的低速验证步骤见
[`docs/replay_validation_notes_zh.md`](docs/replay_validation_notes_zh.md)。
