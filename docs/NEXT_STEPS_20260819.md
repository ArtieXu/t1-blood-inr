# 未解决问题清单 + 下一步顺序

生成：2026-08-19
配套阅读：`tools/docs/RANK5_AUDIT_20260819.md`（每条的证据与推导）

**当前一句话状态**：环境已 setup 但**在 5070 上零执行证据**；科学停止 gate `RANK5_MODEL_GAP_DETECTED` 未变，且本次已把它从「观察」升级为「可证明」。

---

## 第一部分：问题清单

### A 类 — 阻塞 Gate B，动手前必须清掉（3 项）

| # | 问题 | 状态 | 处理 |
|---|---|---|---|
| A1 | `utils.py` 依赖 `torchvision` + `scikit-image`，但 pip 安装行、`verify_migration --env`、文档「已验证依赖」三处都没有。环境会全绿然后 smoke test 在 `import utils` 崩 | **未验证**（不知道你机器上装没装） | 跑 `doctor.sh`，10 秒出结果。缺了就从 cu128 源装 `torchvision`，pip 装 `scikit-image` |
| A2 | CUDA 路径没写进 shell 配置，重开 WSL 就丢 | 已给方案，**未验证** | `source tools/env.sh`；确认可用后挂进 `~/.bashrc` |
| A3 | 三个 `.sh` 经 OneDrive/Windows 编辑器可能变成 CRLF，bash 会报 `\r: command not found` | **未验证** | `dos2unix tools/*.sh` 或 `sed -i 's/\r$//'` |

### B 类 — 文档与代码不符，需回写进 `SETTING_PROGRESS...md`（5 项）

| # | 问题 | 证据等级 |
|---|---|---|
| B1 | 文档要求「使用新的 5070 输出目录」，但 `train_inr_unsup_spiral.py:411` 的 `log_path='./log/...'` 是**相对 CWD** 的，脚本没有 `--out_dir`。带时间戳所以不覆盖旧结果，但落点取决于启动目录 | 静态 |
| B2 | 文档把 TI 63 vs 50 当作实验变量，但在 `--time_coords ti_span` 下 min-max 归一化把 `TI_START_MS` **精确消掉**；且三参数 IR 拟合的 T1 对常数 TI 平移**严格不变**（实测两种 TI 都是 1871.843 ms），只有 Mz/M0 差 1.5%。规则该留（出处 + 反转效率解释），但它**不解释 image gate 失败** | 静态 + CPU 执行 |
| B3 | 文档要求「记录 loss 是否有合理趋势」，但 32 步**不是** 1600 步的前缀 —— `epochs_per_level = epochs//n_levels` 和 `StepLR(step_size=epochs//2)` 都从 `--epochs` 派生。耗时/显存可外推，loss 轨迹不能 | 静态 |
| B4 | `verify_migration.py` 只检 `location=local` 行，manifest 里 4 行 `location=drive` 的 P2 产物**永远不会被验证**。「verify_migration 通过」≠ P2 证据已核对 | 静态 |
| B5 | exact-gate notebook 里 `assert retained_energy > 0.99999` 是**虚假信心**：rank-5 保留字典自身能量 99.9996%，但投影真实解剖时梯度能量丢 16.4%。两者无单调关系 | CPU 执行 |

### C 类 — 科学 blocker（当前主线）

| # | 问题 | 状态 |
|---|---|---|
| C1 | 硬 rank-5 的 **oracle 投影天花板本身**过不了 image gate（grad 16.42% vs 阈值 15%）。任何算法都超不过天花板 → 失败在 prior 表达力，不在优化器 | **已证明**（本次 CPU 复算，且精确复现了 `MIGRATION_5070.md` 的 4.32/16.41/19.36） |
| C2 | 加大物理字典的秩基本无效：K=5→8 grad 只降 7%，K=10 才勉强压线 | **已量化排除** |
| C3 | 扩 Mz 网格（−1..+1）或 T1 网格（100–8000）**完全无效**，grad 变化在第 4 位小数 | **已量化排除**（`PROGRESS_AND_OUTLOOK` §5 可把这一维从待扫参数表划掉） |
| C4 | 修订 prior 的**具体形态还没定**：软约束 vs 物理 K=5 + 残差模态 vs 换秩 | **待决策 — 需要你拍板** |
| C5 | 外壳 k 残差（\|k\| 0.75–1.0）从 0.341 到 0.299，**换任何温度基都基本不动**。这是空间侧问题：`A∘Φ` 条件数 / B0 / 轨迹前向失配 | **已定位，未解决** |
| C6 | B0/off-resonance **不在训练前向算子里**（只在重建后做 deblur） | 未开始（P3） |
| C7 | `A∘Φ` 的 TPSF / 条件数诊断**从未测量和保存**（P0 交付物） | 未开始 |
| C8 | MATLAB/Python projector parity（P0 交付物）未做 | 未开始 |
| C9 | 没有可复用的 coefficient-domain trainer：`build_t1_basis.py`、`train_coeff_inr_spiral.py` **都不存在**，只有 notebook 里的 inline 原型 | 未开始 |
| C10 | `run_coeff_subspace_exact_gate.ipynb` **从未执行**（execution_count 全为 None），且是 Colab 专用（cell 2/4 会直接崩）。另外它的 phantom 按构造精确落在 rank-5 内，**测的不是当前 blocker** | 需移植 |

### D 类 — 证据缺口（不阻塞，但影响可信度）

| # | 缺口 |
|---|---|
| D1 | **5070 上零执行证据** —— 训练、smoke test 一次都没跑过 |
| D2 | Drive 上的 P2 执行产物（result folder / summary CSV / decision JSON / GIF）**全部未验证** |
| ~~D3~~ | ~~data-derived rank 口径差异~~ **已关闭**：是我漏了 `highpass<0.15`。补上后数据驱动 K=6 不过、K=7 过，与文档一致。P2 的 target / TI / 字典 / 投影定义已逐项核对，与本次审计可比 |
| D4 | `full_spiral_reference.mat`（72 MB）本次传输超时未取。D3 已由别的途径关闭，这条降级为「有空再取」 |
| **D8** | **`gradient_NRMSE<0.15` 和 `highpass_NRMSE<0.15` 没有任何推导**，只硬编码在两个 notebook 里；`PROGRESS_AND_OUTLOOK` §7 对这条只有定性表述。且**两个 notebook 的 gate 互不一致**（exact-gate 缺 highpass 和 off_edge_ringing，见报告 §4.3/§4.8） |
| D5 | 旧主机上 `run_inr_unsup_spiral_v2.ipynb` 里的 W&B token **未 revoke**（该文件正确地没有被迁移过来，但源头还在） |
| D6 | 严格 matched B0 / shared-mask 的 MATLAB 评估未对最新 runs 完成 |
| D7 | GASSP1/GASSP2 重复性、三 seed 稳定性未建立 |

---

## 第二部分：下一步顺序

### Step 0 — Gate A：环境复核（约 10 分钟，无风险）

```bash
cd ~/T1_blood_INR_5070_migration_20260817
dos2unix tools/*.sh          # 或 sed -i 's/\r$//' tools/*.sh
source tools/env.sh
bash   tools/doctor.sh
```

清掉 A1 / A2 / A3。**通过 = 环境可用，不代表任何实验跑过。**

### Step 1 — Gate B：32 步冒烟（约 20 分钟）

```bash
bash tools/smoke.sh            # U0，脚本自带默认参数
ARM=U1 bash tools/smoke.sh     # backward 多一次伴随算子，显存上限压力测试
```

要拿到的数字：runtime signature、每步耗时、**峰值显存**（12 GB 卡）、**系统内存峰值**（WSL 只有 7.3 GiB，在 `host_resources.txt` 里）、输出形状与有限性。
`check_run.py` 会自动判定并明确写出「这是 runtime 结果，不是收敛/图像质量/科学结论」。

关掉 D1。

### Step 2 — 回写文档（半小时，纯文字）

把 B1–B5 五条写进 `SETTING_PROGRESS_AND_TEST_PLAN_20260819.md`。
**这一步别跳过** —— B2 和 B5 会直接影响下一个人怎么设计实验（一个会去做无意义的 TI 重跑，一个会拿 `retained_energy` 当准入指标）。

### Step 3 — 决策点：定下修订 prior 的形态 ⬅️ **需要你拍板，我做不了**

`tools/probes/subspace_preb0_ti50.csv` 里的天花板数据。
**gate 用 P2 的三项 image-domain 标准：mag<0.05、grad<0.15、hp<0.15**（`build_p2_..._notebook.py:474` 的 `GATES`）。

| 候选 | mag | grad | hp | 过 gate | 物理 T1 成分 |
|---|---:|---:|---:|:--|---|
| 物理 K=5（当前） | 0.0432 | 0.1642 | 0.1936 | ✗ | 完整 |
| 物理 K=10 | 0.0347 | 0.1487 | 0.1764 | ✗ | 完整但秩翻倍 |
| 物理 K=5 + 2 残差模态 | 0.0344 | 0.1361 | 0.1615 | ✗ | 完整保留 |
| 物理 K=5 + 3 残差模态 | 0.0302 | 0.1288 | 0.1531 | ✗ | 完整保留 |
| **物理 K=5 + 4 残差模态** | 0.0276 | 0.1210 | **0.1445** | **✓ 最小可行** | **完整保留** |
| 数据驱动 SVD K=7 | 0.0266 | 0.1210 | 0.1455 | ✓ | 丢失物理解释 |

> **2026-08-19 修订**：本文件第一版把「+2 残差模态」列为可行，那是漏了 `highpass<0.15`（我误用了 exact-gate notebook 的 gate，它没有这一条）。补上之后最小可行是 **+4**。这同时也解释并关闭了 D3——`MIGRATION_5070.md` 说的「data-derived rank 7 passes」是对的。

建议 **物理 K=5 + 4 个残差模态**（总秩 9）：与数据驱动 K=7 几乎打平，代价是多 2 个自由度，换来 5 个物理 T1 基向量完整保留、T1 定量解释不受影响，且符合 `AGENTS.md:38` 许可的 "small residual dynamic subspace"。

**但先看清两条前提：**

1. **表里「残差模态」和「数据驱动」两组的基是从 target 算出来的，是 oracle。** 实际重建没有 target，这 4 个模态得从别处估，估计误差直接进 prior → 0.1445 是**上界的上界**。物理 K 那几行不同，它们的基与数据无关，天花板是真能达到的。
2. **阈值 0.15 本身没有推导**（见报告 §4.8）。「rank-5 不够」这个结论不依赖它（hp 超了 29%），但「+2 还是 +4」完全取决于它——阈值挪到 0.16，答案就变了。

所以这个决策点实际是**三个问题**：

- **(a) 先给 0.15 一个锚点。** 最有说服力的是重复性地板：GASSP1 vs GASSP2 同一受试者两次扫描之间的 grad/hp NRMSE。低于这个数的差异不可重复，卡再严没意义。
- **(b) 残差模态从哪估？** fully sampled reference（解剖泄漏，评估必须换受试者）／ 实测数据自监督（干净但更难）／ 扩展物理字典（加 B0 相位演化、流动、部分容积项——仍是物理基，无泄漏，但要先知道缺的是什么）。
- **(c) 硬约束还是软约束？** 正交补投影 vs `λ‖(I−ΦΦᵀ)X‖` 罚项。

**(a) 和 (b) 定不下来，Step 5 之后全都没法开始。**

### Step 4 — 补 P0：`A∘Φ` 条件数 + projector parity（关 C7 / C8）

Step 3 定了形态之后做，因为条件数要针对最终的 Φ 算。
交付物：分 \|k\| 壳的 TPSF / 条件数诊断（**保存下来**，`PROGRESS_AND_OUTLOOK` §10.1 点名这项从没存过），以及 MATLAB/Python 同一 projector 的数值一致性。

### Step 5 — 移植 exact gate notebook 到 WSL（关 C10）

只需改 cell 2 和 cell 4（本地路径 + 去掉 `!pip install`），其余逻辑与平台无关。
**注意它验证的是 `A∘Φ` 条件数与 INR 优化，不是 prior 的表达力** —— 它的 phantom 按构造精确落在子空间内。跑通它**不解除** `RANK5_MODEL_GAP_DETECTED`。
另外 notebook metadata 写的是 Colab A100，换到 5070 是一条新的 runtime 分支，所有对照臂都要在同一栈上重跑。

### Step 6 — P2 retrospective 用修订 prior 重跑（真正解除 gate D）

全部在同一 5070 栈上，包括对照臂。旧 T4 结果只作历史参考，不能混进单变量比较。
顺手关掉 D3/D4：`probes/subspace.py --target results/full_spiral_reference.mat --target_var I_FS_an`。

### Step 7 — 才轮到 Gate E

`AGENTS.md` 的锁：**revised prior 通过独立图像 gate 之前，不得开始 measured-data tuning。** 之后还必须先做同一 5070 栈上的 matched short parity，再考虑 800/1600-step 研究。

---

## 可以并行做的（不占主线）

- **D5 / 现在就做**：去旧主机 revoke 那个 W&B token。迁移目录里已扫过，无凭据泄漏，但源头还在。
- **D2**：登录正确的 Google 账号，核对 P2 四个产物的 file ID、大小、SHA-256，写回 manifest。
- **C9**：把 notebook 里的 inline 原型抽成 `build_t1_basis.py` + `train_coeff_inr_spiral.py`。Step 3 定了形态之后再抽，否则要返工。

---

## 一条不能忘的边界

Step 3 表里「过 gate」指的是 **oracle 投影天花板**过线 —— 这是**必要条件，不是充分条件**。
C5 说得很清楚：外壳 k 残差换任何温度基都不动。换 prior 只解开第一道锁，`A∘Φ` 条件数和 B0 前向失配是另外两道，得靠 Step 4 和 P3 分别处理。
