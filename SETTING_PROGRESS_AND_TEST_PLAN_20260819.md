# T1 Blood INR 动态螺旋 MRI

## 5070 环境接管进度与测试计划

更新时间：2026-08-19

## 当前结论

当前状态为：**环境 setup 已完成并通过验证，但测试尚未开始**。

- 尚未启动训练。
- 尚未产生新的收敛结果、图像质量结果或科学解释。
- 当前科学停止 gate（门槛）仍为 `RANK5_MODEL_GAP_DETECTED`。
- 在修订后的 prior（先验模型）通过独立图像域 gate 之前，不得开始 measured-data tuning（实测数据调参）。

## 1. Setting 进度

| 项目 | 状态 | 具体信息 |
|---|---|---|
| WSL2（Windows Linux 子系统第 2 代） | 已配置 | Ubuntu 22.04.5 LTS |
| Python 虚拟环境 | 已配置 | `/home/t1user/.venvs/t1_5070` |
| Python | 已验证 | 3.10.12 |
| GPU | 已验证 | NVIDIA GeForce RTX 5070 |
| GPU 计算能力 | 已验证 | Compute Capability 12.0 |
| 显存 | 已验证 | 约 11.94 GiB 可见 |
| NVIDIA 驱动 | 已验证 | 576.88 |
| CUDA（GPU 加速平台） | 已验证 | 运行时 12.8；`nvcc` 12.8.93 |
| PyTorch | 已验证 | `2.7.1+cu128` |
| torchkbnufft | 已验证 | `1.5.2` |
| tiny-cuda-nn（CUDA 神经网络扩展） | 已验证 | `2.0`，锁定 commit `749dd70c5afc5a9dadb85e5652ed65d55e0ba187` |
| 关键 Python 依赖 | 已验证 | NumPy 2.2.6、SciPy 1.15.3、Pandas 2.3.3、Matplotlib 3.10.9、h5py 3.16.0 等 |
| 项目本地迁移检查 | 已验证 | `python3 verify_migration.py` 通过 |
| 环境迁移检查 | 已验证 | `python3 verify_migration.py --env` 通过 |
| W&B | 未使用 | 仅安装，未登录、未上传 |
| 32-step smoke test（烟雾测试） | 未执行 | 尚无运行产物 |

补充：WSL 当前可见系统内存约 7.3 GiB。主机物理内存为 16 GB；烟雾测试时仍需观察系统内存和显存峰值。

## 2. Setup 结构

```text
Windows 主机
├─ RTX 5070 / 6700X / 16 GB RAM
├─ WSL2
│  └─ Ubuntu 22.04
│     ├─ /home/t1user/.venvs/t1_5070
│     │  ├─ Python 3.10.12
│     │  ├─ PyTorch 2.7.1 + CUDA 12.8
│     │  ├─ torchkbnufft 1.5.2
│     │  └─ tiny-cuda-nn 2.0
│     └─ /usr/local/cuda-12.8
│        └─ nvcc 12.8.93
└─ 项目目录
   C:\Users\xzc-G\OneDrive\Desktop\UW\
   T1_blood_INR_5070_migration_20260817
```

当前 CUDA 路径变量是在测试会话中设置的，**尚未写入 shell 配置文件持久化**。重开 WSL 后，需要重新激活环境并设置 CUDA 路径；这部分尚未修改。

典型测试会话需要的环境设置为：

```bash
source /home/t1user/.venvs/t1_5070/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
```

## 3. 下一步 testing 顺序

### Gate A：环境复核

在新 WSL 会话中：

1. 激活 `/home/t1user/.venvs/t1_5070`。
2. 设置 CUDA 路径。
3. 再次运行：

   ```bash
   python3 verify_migration.py --env
   ```

此步骤只验证环境，不代表训练已经运行。

### Gate B：32-step 5070 smoke test

获得确认后再执行 32 步烟雾测试，要求：

- 使用新的 5070 输出目录。
- 不覆盖旧 T4 结果。
- 不登录 W&B，不上传文件。
- 固定 entrypoint、数据路径、随机种子和输出位置。
- 记录完整 runtime signature（运行时签名）：GPU、驱动、PyTorch、CUDA、tiny-cuda-nn、Python 版本。
- 记录 loss 是否有限、是否有合理趋势、每步耗时、峰值显存、输出尺寸和输出是否有限。
- 出现非有限值、显存不足或输出异常时立即停止。

仓库中的 `train_inr_unsup_spiral.py` 默认是较长训练入口，并使用当前实测采集的 `TI = 63 + 200*n ms`。因此在启动前必须先确认 entrypoint 与数据契约，不能把一次 32 步运行误称为科学实验或收敛结果。

### Gate C：匹配短 parity

只有 smoke test 通过后，才进行同一 5070 软件栈上的短 parity（匹配对照实验）：

- 数据、轨迹、DCF（密度补偿函数）、随机种子、步数、checkpoint 选择规则、软件和硬件保持一致。
- 旧 T4 结果只作为历史参考，不能与 5070 结果混合成严格单变量比较。

### Gate D：修订 prior 的独立图像 gate

当前科学 blocker 是 `RANK5_MODEL_GAP_DETECTED`：硬 rank-5 物理 T1 子空间未通过 oracle image gate（独立图像门槛）。下一项科学工作应先检查：

```text
notebooks/current/run_coeff_subspace_exact_gate.ipynb
```

修订 prior 时应保留主 T1 成分。低 k-space DC（数据一致性）损失不能证明图像恢复正确，也不能证明结果唯一。

### Gate E：实测数据调参与正式实验

仅当 revised prior 通过独立图像 gate 后，才允许开始 measured-data tuning。正式实验前还必须完成同一 5070 栈上的匹配短实验，之后才考虑 800/1600-step 研究。

TI 约定必须严格区分：

- 回顾性参考实验：`TI = 50 + 200*n ms`
- 当前实测采集：`TI = 63 + 200*n ms`

## 4. 证据等级分离

| 证据层级 | 当前状态 |
|---|---|
| 源码/静态检查 | 已完成部分迁移检查；入口和 notebook 已识别 |
| 真实执行 | 环境检查已执行；训练和 smoke test 尚未执行 |
| 目标收敛 | 未验证 |
| 图像质量门槛 | 当前 rank-5 prior 未通过独立 gate |
| 科学解释 | 受 `RANK5_MODEL_GAP_DETECTED` 限制，暂不能进入实测调参 |

## 5. 当前允许的最小行动清单

1. 新 WSL 会话中复核环境。
2. 确认 smoke test 的 entrypoint、数据路径、输出目录、随机种子和 TI 契约。
3. 运行一次 32-step 5070 smoke test。
4. 保存 runtime signature、loss、耗时、显存和输出有限性记录。
5. smoke test 通过后，再决定是否进入 5070 parity。

在执行任何修改、安装、训练、上传或覆盖结果之前，等待项目负责人确认。

