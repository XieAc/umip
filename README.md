# umip：自制 UMI 类夹爪的采集、训练与真机部署

手持 XAC 夹爪（iPhone Stray Scanner 位姿 + D435i 腕部相机 + 两个 XC330 编码器）采集演示，
训练 Diffusion Policy，在斜装 45° 的 UR5 上用同构执行夹爪真机执行"抓方块放进碗"。

| 目录 | 运行在 | 内容 |
|---|---|---|
| [`collection/`](collection/) | 采集电脑（Windows） | 夹爪宽度标定、T_I_G 标定、演示采集、同步、构建 episode、导出 Zarr、UR5 轨迹重播 |
| [`training/`](training/) | PRO6000 / RTX 5090（Linux） | v2 训练配置 `train_xac_v2.yaml`、训练与离线评估 |
| [`deployment/`](deployment/) | RTX 3060 + UR5（Linux） | 策略推理与执行（软地板、夹爪开关模式）、UR5 TCP 标定、夹爪/相机驱动 |

## 数据流

```text
collection/  采集 (record_435i + gripper_grasp_control) → prepare → build_dataset → export_umi_zarr
             └─ 输出 dataset_v3/replay.zarr.zip + replay_manifest.csv/json
training/    放到 training/dataset_v3/ → bash check_dataset.sh → bash train_v2.sh ... → eval_offline.py
             └─ 输出 data/outputs/<run>/checkpoints/best.ckpt
deployment/  复制到 deployment/data_model/<run>/checkpoints/ → test/test_model_inference.py
```

## 不在仓库里的东西

- 原始采集数据、构建的 episode、`dataset_v*/` Zarr、训练输出和 checkpoint、运行日志：体积大，另行保存。
- 旧版本：早期错误的训练方案（plateau、冻结编码器）、有 10° 误差的旧 T_I_G、TIG 标定的原始 session、
  各种 `.bak` 备份和一次性分析/修复脚本都没有放进来。

## 标定（已放入仓库的当前版本）

| 文件 | 用途 |
|---|---|
| `collection/calibration/T_I_G.npy`、`T_I_G.json` | iPhone→夹爪外参，2026-10-24 按"夹爪竖直时 iPhone 水平"修正 10°（`level_mount_correction_20261024`） |
| `*/calibration/gripper_width.json` | 编码器 tick → 两指尖总开口宽度（11 点实测） |
| `deployment/calibration/tcp_gripper_01_gx_down_candidate.json` | UR5 法兰→G 的 TCP，部署实际使用 |
| `deployment/calibration/tcp_gripper_01.json`、`tcp_samples_01.json`、`probe_position_01.json` | TCP 原始拟合及其样本，部分脚本和测试用作默认值 |

换夹爪、换相机安装或换机械臂后需要重新标定，步骤见各目录 README。

## 快速入口

- 采集：`collection/README.md`（Windows，`python deploy.py check --project` 自检）
- 训练：`training/README.md`
- 真机部署：`deployment/docs/model_xac_v3_h1img1_800ep_20261025.md`（当前模型的完整上机流程、参数和 2026-10-29 故障记录），其余见 `deployment/README.md`

## 测试

```bash
cd collection && python -m unittest test_collection_deployment test_contact_variance test_dataset_preparation test_replay_deployment test_replay_xac_ur5 test_shake_sync test_shake_timing
cd training   && python -m unittest test_xac_training test_obs_encoder
cd deployment && python -m unittest test_xc330_switch test_policy_control test_policy_loading test_replay_xac_ur5 test_verify_tcp_transform test_calibrate_robot_tcp
```

整理时（2026-10-29）三组测试分别为 21、8、54 项，全部通过。

## 许可证

`training/` 和 `deployment/` 中的 `diffusion_policy/`、`umi/` 改编自
[UMI](https://github.com/real-stanford/universal_manipulation_interface)（MIT，见 `LICENSE_UMI`）。
其余代码的许可证由作者决定。
