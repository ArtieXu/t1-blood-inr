# `tools/` — Gate A / Gate B 可执行包

放在项目根目录下，不修改任何既有代码。全部脚本假定项目根 = 本目录的上一级。

## 文件

| 文件 | 作用 |
|---|---|
| `env.sh` | **source 用**。激活 venv + 设置 CUDA 路径，幂等，可挂进 `~/.bashrc` |
| `doctor.sh` | Gate A：环境复核（不训练） |
| `deps.py` | 补 `verify_migration.py --env` 没覆盖的 import（torchvision / scikit-image 等） |
| `sysinfo.py` | 记录完整 runtime signature 为 JSON |
| `smoke.sh` | Gate B：一次 32-step 冒烟测试，参数固定 |
| `check_run.py` | Gate B 验收：loss 有限性 / 趋势 / 每步耗时 / 显存峰值 / 输出形状与有限性 |
| `probes/subspace.py` | rank-K temporal prior 的 oracle 投影上限审计（纯 CPU） |
| `docs/RANK5_AUDIT_20260819.md` | 本次复核报告（先读这个） |

## 用法

```bash
cd ~/T1_blood_INR_5070_migration_20260817        # 改成你的实际路径

# --- Gate A：新 WSL 会话里的环境复核 ---
source tools/env.sh
bash   tools/doctor.sh
# 产物：runs/gate_a_<时间戳>/{gate_a.log, sysinfo.json}

# --- Gate B：32 步冒烟测试（Gate A 通过后再跑）---
bash tools/smoke.sh            # 默认 arm U0（脚本自带默认值）
ARM=U1 bash tools/smoke.sh     # backward 里多一次伴随算子，显存压力最大
# 产物：runs/smoke_<ARM>_seed0_<时间戳>/
#         ├── sysinfo.json
#         ├── nvidia_smi_before.txt / nvidia_smi_after.txt
#         ├── host_resources.txt        （/usr/bin/time -v，含系统内存峰值）
#         ├── train_stdout.log
#         ├── acceptance.txt
#         └── log/smoke32_5070_<ARM>_<时间戳>/   ← 训练脚本自己的输出
```

想让 CUDA 路径在每个新 WSL 会话自动生效：

```bash
echo 'source "$HOME/T1_blood_INR_5070_migration_20260817/tools/env.sh"' >> ~/.bashrc
```

（路径改成你的实际位置。venv 或 CUDA 位置不同时，可先 `export T1_VENV=... T1_CUDA_HOME=...` 再 source。）

## 输出位置

所有产物写进项目根下的 `runs/`，用 `T1_OUT_ROOT=/其他/路径` 可改。
**不会写入任何既有的 T4 结果目录。**

原因见报告 §2.1：`train_inr_unsup_spiral.py` 的 `log_path = './log/...'` 是相对**当前工作目录**的，脚本没有 `--out_dir`。`smoke.sh` 通过先 `cd` 到目标目录来固定输出位置，没有改训练脚本本身（它在 manifest 里有 SHA-256）。

## rank-5 审计

```bash
python3 tools/probes/subspace.py            # 默认 preb0 target，TI = 50 + 200n
python3 tools/probes/subspace.py \
        --target results/full_spiral_reference.mat --target_var I_FS_an \
        --ranks 4 5 6 7 8 --out audit_reference
```

纯 numpy/scipy，CPU 即可，不碰 GPU、不联网、只读输入。
它回答的是：**给定一个硬 rank-K 温度子空间，任何重建方法的最好结果是什么。** 结果见报告 §4。

## 这套脚本不做什么

- 不改任何既有文件（唯一例外见报告 §5）
- 不训练超过 32 步
- 不联网、不上传、不登录 W&B（`env.sh` 强制 `WANDB_MODE=disabled`）
- 不产生任何科学结论。Gate B 通过 = 这套软件栈在这块卡上**能跑**，仅此而已。
