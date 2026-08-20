# RTX 5070 migration handoff

Last updated: 2026-08-17

Project root on the old host:
`/ibic/projects3/ORCA/zechenxu/INR_for_DynamicMRI/T1_blood_INR`

Target hardware: NVIDIA GeForce RTX 5070, 12 GB VRAM; 16 GB system RAM.
The current directory is not a Git repository, so there is no commit hash to
use as provenance. File SHA-256 values in `MIGRATION_5070_MANIFEST.tsv` are the
transfer integrity record.

## One-paragraph project summary

This project reconstructs dynamic spiral MRI for IJV blood T1 estimation using
MATLAB/Python operator parity, CG baselines, and implicit neural representation
(INR) models. MATLAB/Python 59-frame CG parity and downstream B0-deblurred T1
agreement are complete. The current research direction is coefficient-domain
`X = Phi*C` reconstruction with a physical T1 temporal prior. The newest
verified P2 retrospective experiment found that rank-5 T1 is helpful relative
to a no-T1 flexible INR, but the hard rank-5 subspace itself fails the oracle
image gate. Therefore measured-data tuning is currently stopped while a softer
T1 constraint or small residual dynamic subspace is evaluated.

## Current evidence

The P2 result folder was verified on Google Drive on 2026-08-16. Both DCF stages
completed 1600 steps on a Tesla T4 and used each arm's own k-space objective for
checkpoint selection.

| Arm | Data NRMSE | Magnitude NRMSE | Gradient NRMSE | High-pass NRMSE | IJV curve NRMSE | Common image gate |
|---|---:|---:|---:|---:|---:|---|
| rank-5 oracle projection | 2.31% | 4.32% | 16.41% | 19.36% | 1.91% | failed |
| R5 T1 + DCF INR | 2.20% | 6.35% | 30.87% | 35.51% | 2.14% | failed |
| F0 no-T1 + DCF INR | 0.14% | 15.95% | 70.77% | 83.21% | 14.27% | failed |

Decision: `RANK5_MODEL_GAP_DETECTED`.

Interpretation boundaries:

- F0 is the clearest evidence that low k-space error does not imply a correct
  image.
- R5 is better than F0 for image and IJV-curve fidelity, but the comparison is
  not parameter-count matched: 358,912 versus 50,298,960 parameters.
- The physical T1 model remains plausible for the mean IJV curve: T1 is about
  1880.8 ms with R-squared about 0.9957.
- Data-derived rank 7 passes the three image-domain gates, whereas merely using
  a rank-7 physical T1 dictionary still fails. The problem includes both too
  few temporal degrees of freedom and prior-shape mismatch.
- The P2 experiment uses synthetic GASSP1 k-space generated from a fully
  sampled pre-B0 reference. It is not a measured-data result.
- `notebooks/current/run_coeff_subspace_exact_gate.ipynb` has static/math
  validation evidence; confirm an executed GPU artifact before calling its
  scientific gate passed.

## What to give the new host

Copy this entire four-file handoff set first:

- `AGENTS.md`
- `MIGRATION_5070.md`
- `MIGRATION_5070_MANIFEST.tsv`
- `verify_migration.py`

Then copy every row marked `required=yes` and `location=local` in the manifest,
preserving its relative path. This is the minimum runnable/evaluable subset.
Copy the rest of the project only when older validation notebooks, figures, or
debug provenance are needed.

Do not transfer the project as one unsanitized archive yet:
`notebooks/current/run_inr_unsup_spiral_v2.ipynb` contains a credential-like
W&B token string. Revoke/rotate the token and remove it from the notebook first.

## Drive sources

These links are needed because the newest executed P2 outputs are not stored in
the local project tree:

- P2 result folder, authoritative executed output:
  <https://drive.google.com/drive/folders/1JqNUB2-tIH5UK01WJnFAWCl11rbybtb6>
- P2 summary CSV:
  <https://drive.google.com/file/d/12WkJuc52QwFfzlpjhMyG_SNKdGyWBYDW/view>
- P2 decision JSON:
  <https://drive.google.com/file/d/10N5LLlA2JuSaQnVsOQsIYjAMxvbUjVHe/view>
- P2 comparison GIF:
  <https://drive.google.com/file/d/1_-FZ1IznL5S-Go8eIUCNFfTGMh9xtxDD/view>
- Uploaded P2 notebook source:
  <https://drive.google.com/file/d/1ivaJCyBlHMy19_E9zwMEhGzfJ4KFVhJU/view>
- `T1_blood_INR_v2/01_current` Drive folder:
  <https://drive.google.com/drive/folders/10sTmS1ttpm03_xrNokqiyxbtlgwHRo3G>

A link does not grant access by itself. Use the correct Google account or copy
the downloaded artifacts into a new local `results/` directory and record their
file IDs, sizes, and SHA-256 values.

## 5070 environment baseline

RTX 5070 is NVIDIA Blackwell compute capability 12.0. Use Linux or WSL2 and a
PyTorch build with CUDA 12.8 or newer. PyTorch 2.7.1 + cu128 is the minimum
known-compatible reproducible baseline, not a claim that it is the newest
release:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install torchkbnufft==1.5.2 ninja pandas scipy matplotlib pillow h5py tqdm imageio scikit-image wandb jupyter
```

Correction 2026-08-19: `torchvision` and `scikit-image` were missing from the
line above. `utils.py` imports `torchvision.utils.make_grid` and
`skimage.metrics`, and `train_inr_unsup_spiral.py` imports `utils`, so a run
fails with `ImportError` in its first second without them. `verify_migration.py
--env` does not cover either package; `_5070_gate_kit/preflight_imports.py`
does. Install `torchvision` from the same cu128 index as `torch` so their CUDA
builds match.

The locked T4 notebooks install tiny-cuda-nn commit
`749dd70c5afc5a9dadb85e5652ed65d55e0ba187` and some explicitly reject any GPU
whose name does not contain `T4`. Do not silently delete the assertion and call
the run equivalent. First try compiling the locked commit on the 5070; if it
fails, create a separate 5070 environment using a Blackwell-compatible
tiny-cuda-nn revision and record the exact commit. A changed tiny-cuda-nn build
defines a new runtime branch and requires matched reruns of every compared arm.

Sixteen GB of system RAM is enough to start, but 32 GB is recommended for WSL,
Jupyter, MATLAB, CUDA compilation, and training at the same time.

Official compatibility references:

- PyTorch Blackwell/CUDA 12.8 support:
  <https://pytorch.org/blog/pytorch-2-7/>
- NVIDIA compute capability table:
  <https://developer.nvidia.com/cuda/gpus>
- Codex project/file behavior:
  <https://learn.chatgpt.com/docs/projects>
- Codex `AGENTS.md` discovery:
  <https://learn.chatgpt.com/docs/agent-configuration/agents-md>

## First checks on the new host

From this project root:

```bash
python3 verify_migration.py
python3 verify_migration.py --env
```

The first command verifies required local files against the manifest. The
second also verifies that CUDA is visible, the GPU is an RTX 5070-class
Blackwell device, PyTorch uses CUDA 12.8 or newer, and required Python packages
import.

Next run only a 32-step smoke test. Record:

- complete runtime signature;
- loss finiteness and trend;
- seconds per step;
- peak GPU memory;
- output shape and finite-value checks.

Only after that passes should a matched short T4-versus-5070 numerical audit or
a full 800/1600-step 5070 experiment begin.

## Prompt for the new host's ChatGPT/Codex

```text
你将接管 T1_blood_INR 动态螺旋 MRI 项目。新主机配置为 RTX 5070
12 GB 显存、6700X 处理器、16 GB 系统内存。你的第一项任务只是完成
“迁移接管审计”，确认资料、环境和当前科学结论；暂时不要修改文件、安装
依赖、启动训练、上传文件或覆盖既有结果。

项目根目录下，依次完整阅读：
1. AGENTS.md
2. MIGRATION_5070.md
3. MIGRATION_5070_MANIFEST.tsv
4. PROJECT_INDEX.md
5. PROGRESS_AND_OUTLOOK_20260811.md

阅读后运行：
    python3 verify_migration.py

如果新主机已经装好项目环境，再运行：
    python3 verify_migration.py --env

不要为了通过第二项检查而自行安装或升级软件。若当前会话能够访问 Google
Drive，则按照 MIGRATION_5070.md 和 manifest 中的链接核对 P2 执行产物；若
没有连接、权限不足或文件不可读，请明确报告“未验证”，不要把链接存在当作
文件已经验证，也不要要求我重复提供文档中已有的 Drive 地址。
如果当前 ChatGPT 会话不能读取本地文件或执行命令，也要如实报告能力限制，
不得根据文件名或提示词猜测检查已经通过。

请用中文给出一份简洁的接管报告，并严格按以下结构输出：
1. 结论：是否已具备开始 5070 冒烟测试的条件（是 / 否 / 有条件）。
2. 本地文件：完整、缺失或哈希不匹配的项目；列出准确路径。
3. 运行环境：GPU、显存、驱动、PyTorch、CUDA、tiny-cuda-nn 和关键依赖；
   每项标记“已验证 / 未验证 / 不满足”。
4. 外部证据：逐项列出已实际读取和未能读取的 Drive 产物。
5. 科研状态：已完成的 gate、当前 blocker，以及允许执行的下一个 gate。
6. 下一步：给出最小行动清单，并在执行任何修改、安装或训练前等待我确认。

必须遵守以下边界：
- 把“源码或静态检查”“真实执行”“收敛情况”“图像质量门槛”和“科学解释”
  分开报告；没有执行产物时不得声称实验已经运行。
- 低 k-space 数据一致性（DC）损失不等于图像恢复正确，也不证明结果唯一。
- 当前停止结论是 RANK5_MODEL_GAP_DETECTED：在修订后的 prior 通过独立图像
  gate 之前，不得开始 measured-data tuning。
- 回顾性参考实验使用 TI = 50 + 200*n ms；当前实测采集使用
  TI = 63 + 200*n ms，不得混用。
- 旧 T4 结果与新 5070 结果必须分开标注。若 PyTorch、CUDA 或
  tiny-cuda-nn 版本改变，严格比较时必须在同一软件栈上重跑所有对照组。
- 不得显示、复制或提交任何凭据、API key、访问令牌或已知 W&B token。
- 技术名词或缩写第一次出现时，用一句中文解释。
```
