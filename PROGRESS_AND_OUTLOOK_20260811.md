# T1 blood INR reconstruction: progress, diagnosis, and outlook

**Recorded:** 2026-08-11  
**Scope:** GASSP1/GASSP2 IJV T1 reconstruction with a fixed acquisition  
**Status:** operator parity is established; the first direct coefficient-domain exact-gate notebook has been implemented but is unexecuted; no validation gate has passed yet

## 中文摘要

- **已完成：** MATLAB/Python spiral operator 与 IJV T1 parity gate 已通过；target-free flexible INR、DCF placement 和 hard rank-4 temporal subspace 均已有完整诊断结果。
- **当前结论：** flexible INR 可以在极低 DC 下得到错误图像；rank-4 prior 可以恢复平滑且数值合理的 IJV curve，却不能保证正确的空间解剖。两者都不是最终成功。
- **高频判断：** 当前细节缺失同时包含 outer-k under-fitting、B0/trajectory 等 forward mismatch，以及 `A*Phi` 的病态或 null-space。Temporal TV 或少量全局 residual modes 均不足以单独解决。
- **下一主线：** 按 low-rank T1 mapping 的 dictionary-normalization-SVD 核心构建 `Phi`，让 INR 直接输出 complex coefficient maps，并以 `X=Phi*C` 进入完整 spiral forward model。
- **执行顺序：** basis/operator contract -> exact synthetic gate -> fully sampled retrospective gate -> B0-aware real reconstruction -> coefficient-domain wavelet/LLR/multiscale prior -> held-out outer-k、GASSP repeat、image quality 与 T1 联合验收。
- **当前停止条件：** 现有 `run_subspace_validation_minimal.ipynb` 不应原样运行；direct coefficient model 通过 synthetic 和 retrospective gates 之前，不启动 real-data seed sweep。

## 1. Project objective and scientific boundary

The project objective is:

> 在遵循采集数据、信号模型和可辨识性边界的基础上，通过 reconstruction 尽可能还原最佳 image quality，并同时保持可信的 T1 定量。

This means that image sharpness, k-space consistency, temporal/T1 behavior, and repeatability are joint requirements. A visually sharpened result is not accepted when held-out measurements or a repeat acquisition do not support it. Conversely, a plausible IJV T1 value alone does not establish that the reconstructed anatomy is correct.

The acquisition is fixed. Reconstruction can recover frequencies that are encoded but poorly conditioned or poorly fitted. Frequencies in the true null space cannot be uniquely recovered from the current measurements; a prior can select a plausible solution, but that content must not be described as measurement-proven detail.

## 2. Current acquisition and protocol contract

The current data path is:

```text
export_gassp1.m
    -> gassp1_data.mat
    -> load_spiral.py
    -> spiral_nufft.py
    -> train_inr_unsup_spiral.py
    -> matched image/T1 evaluation
```

Relevant protocol facts from the [ExamCard](../../JHU/T1_Blood/20240926_ML/DICOM/IJV_T1_0926.ExamCard.html) and raw files are:

- Act. TR/TE: approximately 200/2.1 ms.
- Golden-angle rotation: 137.51 degrees.
- 55 acquired phases; the established reconstruction selects 50 physical frames.
- One spiral arm per selected TI, 28 coils, spiral-out readout.
- Background suppression is enabled, with a recorded pulse timing of 138 ms.
- The current effective-TI contract is `TI = 63 + 200*n` ms, `n = 0,...,49`. The 200 ms label is a nominal phase interval rather than, by itself, the time since the final preparation pulse.

The acquisition timing is treated as fixed. The reconstruction model may be improved to represent the acquisition physics more faithfully; this does not change the acquisition.

## 3. What has been established

### 3.1 MATLAB/Python spiral operator gate

The gross export/layout/NUFFT failure mode is no longer the primary blocker. The matched B0-deblurred IJV T1 values reported in [HANDOVER.md](HANDOVER.md) are:

| Reconstruction | IJV T1 (ms) | Fit R2 |
|---|---:|---:|
| MATLAB CG | 2004.480 | 0.991344 |
| Python parity CG | 2008.671 | 0.991524 |

The T1 difference is 4.19 ms. Exact complex-image equality is not established, but the clinically relevant operator/T1 parity gate has passed.

### 3.2 Target-free flexible INR remains non-identifiable

The target-free DCF-placement study is recorded in [REVIEW_PACKET_unsup_dcf.md](REVIEW_PACKET_unsup_dcf.md). U2, with DCF in the global objective, was the only anatomically readable arm, but it did not pass the success criteria:

- pre-B0 IJV T1: 1711.6 ms versus 1880.8 ms in the fully sampled reference;
- fit R2: 0.817;
- IJV second-difference roughness: 7.37 times the reference;
- no strict matched B0/shared-mask evaluation was completed.

The same study measured that the first 10% of the readout contains 97.1% of unweighted k-space energy and that structured residual grows by approximately 4.36 times from the center toward the outer readout. Therefore, loss definition and forward-model error strongly affect whether measured high-frequency samples influence training.

### 3.3 Rank-4 temporal prior repairs the curve but not the image

The latest completed comparison is [core_summary.csv](results/temporal_subspace_gate/20260805_224440_628356/core_summary.csv). Its main results are:

| Arm | Uniform DC | Magnitude NRMSE vs CG | pre-B0 IJV T1 (ms) | R2 | IJV d2 / reference |
|---|---:|---:|---:|---:|---:|
| self flexible | 0.00104 | 0.317 | 1648.5 | 0.943 | 2.07 |
| real flexible | 0.01227 | 0.452 | 1506.4 | 0.380 | 9.87 |
| real rank-4 subspace | 0.02925 | 0.787 | 1879.6 | 0.994 | 0.505 |

The result establishes two complementary failure modes:

1. Very low data-consistency loss can coexist with a wrong image and wrong temporal behavior. The one-arm-per-TI inverse problem is highly non-identifiable without a sufficiently informative prior.
2. A hard rank-4 IR subspace can force an excellent IJV recovery curve while suppressing internal anatomy and increasing image/DC error. It is a useful diagnostic, not a successful final reconstruction.

The fixed-window comparison is stored on Drive as [core_fixed_window.gif](https://drive.google.com/file/d/1Afkq-Z0vDoAgiFfUEvd3L6K7gCH13NrK/view), with the associated [core diagnostics](https://drive.google.com/file/d/16SWSYmYwek5IHLt2eROj36ori49lOgB_/view).

### 3.4 The prepared validation notebook is unexecuted and no longer the desired final model

[run_subspace_validation_minimal.ipynb](notebooks/current/run_subspace_validation_minimal.ipynb) contains eight code cells, all currently unexecuted. Its gate order—exact synthetic recovery before real data—is correct, but it should not be run unchanged because it still uses:

- rank 4;
- a 3D `(t,y,x)` HashGrid that predicts 50 frames and then projects them;
- temporal TV on the projected time series;
- no B0/off-resonance term in the forward operator.

There is no active local INR training process at the time of this record.

## 4. Current diagnosis of the missing high-frequency detail

The missing detail should not be attributed only to “insufficient temporal resolution” or to TV. Three mechanisms must be separated:

1. **Recoverable, measured outer-k content is under-fitted.** Center-k energy dominates a conventional global loss, while the current INR and loss do not fit outer samples sufficiently well.
2. **The real forward model is incomplete.** B0/off-resonance, residual trajectory error, coil-map error, flow/cardiac phase, or motion can create structured outer-readout residual. Applying B0 deblurring only after reconstruction does not make the training operator physically matched.
3. **The joint encoding `A*Phi` has a null space or poor conditioning.** In this component, a prior selects among possible images; it cannot prove the true missing anatomy.

A read-only projection audit of the fully sampled reference using a normalized physical rank-5 IR basis found the following complex temporal projection residual by radial k-space shell:

| Normalized radius | Projection residual |
|---|---:|
| 0–0.25 | 0.043 |
| 0.25–0.50 | 0.172 |
| 0.50–0.75 | 0.257 |
| 0.75–1.00 | 0.338 |

After removing the physical rank-5 component, the first three global residual temporal modes explain only approximately 29–31% of the residual energy in the outer two shells. Therefore, adding one to three unconstrained global temporal modes is not expected to recover the missing spatial detail. The stronger next candidate is a local or multiscale spatial prior on the subspace coefficient maps.

These projection figures are diagnostic values from the current reference volume and should be reproduced by a saved analysis script before use in a paper or formal result table.

## 5. Theoretical reconstruction direction

The T1 prior must use the same core construction as the established low-rank T1-mapping method:

```text
protocol-specific signed IR dictionary
    -> L2-normalize every temporal curve
    -> SVD
    -> retain the first K temporal basis functions Phi
```

The signal model is

```text
D(t; T1, Mz) = 1 + (Mz - 1) * exp(-TI(t)/T1).
```

The reference MATLAB implementation is [getT1Prior.m](../../simulation/T1_GASSP_SLR_Simultaion/Simulation_IR/getT1Prior.m). The reconstruction should directly parameterize complex coefficient maps:

```text
C_theta(x,y) -> K complex coefficient maps
X(t,x,y) = sum_k Phi(t,k) * C_theta(k,x,y).
```

This differs materially from predicting all 50 images with a 3D INR and applying `Phi*Phi'` afterward. The direct coefficient formulation makes the physical subspace part of the forward model and reduces the unknowns from 50 complex images to K complex coefficient maps.

### Initial engineering contract

| Parameter | Primary setting | Sensitivity tests |
|---|---|---|
| TI | `63 + 200*n` ms, 50 frames | only change after sequence-timing evidence |
| T1 grid | 200–5000 ms, 1500 points | expand only if reference/fit requires it |
| Initial `Mz/M0` grid | -1 to 0 | blood-focused -1 to -0.8; legacy -1 to 1 |
| Rank | K=5 | K=4 and K=6 |
| Curve preprocessing | per-curve L2 normalization | none |
| Baseline DC | noise-whitened/global complex L2 | uniform versus controlled radial weighting |
| Temporal TV | 0 for the primary coefficient model | diagnostic only |

The construction rule is locked. The Mz range and rank remain engineering parameters to be selected by projection and retrospective-recovery evidence, not by the apparent smoothness of one real-data IJV curve.

## 6. Next-step logic flow

```text
Freeze protocol and T1-prior contract
                |
                v
Audit conditioning/TPSF of the joint operator A*Phi by |k| shell
                |
                v
Implement direct coefficient reconstruction: X = Phi*C
                |
                v
Exact sharp in-subspace synthetic gate
       | fail                         | pass
       v                              v
implementation/loss/optimizer    retrospective full-reference gate
must be fixed first                    |
                    +------------------+------------------+
                    |                  |                  |
          coefficient CG good,  both methods blur,  retrospective good,
          coefficient INR bad    or lose edges       real data bad
                    |                  |                  |
           INR parameterization   A*Phi conditioning  forward mismatch:
           or optimization        or prior problem    B0/trajectory/SMap
                    +------------------+------------------+
                                       |
                                       v
                    B0-aware real-data reconstruction
                                       |
                                       v
            local/multiscale priors on coefficient maps
                                       |
                                       v
       held-out outer-k + full reference + GASSP repeat + T1 validation
```

## 7. How the work should be executed

### Phase P0 — Basis and projector contract

Deliverables:

- one shared MATLAB/Python basis manifest containing TI, T1 grid, Mz grid, rank, normalization, and hashes;
- projector agreement between MATLAB and Python to numerical precision;
- K=4/5/6 projection error in image, gradient, and actual radial k-space shells;
- coefficient-domain TPSF/conditioning diagnostic for `A*Phi`.

Stop if MATLAB and Python do not build the same projector.

### Phase P1 — Exact synthetic gate

Construct a sharp vessel/edge phantom exactly inside the K=5 subspace, simulate data with the existing SMap and GASSP trajectory, and reconstruct from random initialization with no image target and no temporal TV.

Initial pass criteria, fixed before training:

- uniform DC energy ratio below 0.01;
- magnitude NRMSE below 0.05;
- no systematic loss of vessel boundaries in gradient/high-pass metrics;
- no unexplained outer-shell residual increase.

Failure indicates an implementation, loss, capacity, or optimization problem. Real-data training must not start after this failure.

### Phase P2 — Retrospective full-reference gate

Use [full_spiral_reference.mat](results/full_spiral_reference.mat) as the known image series, forward-simulate the fixed one-arm-per-TI GASSP encoding, and compare:

1. direct coefficient least-squares/CG;
2. direct coefficient INR;
3. coefficient reconstruction with a joint wavelet prior;
4. coefficient reconstruction with patch locally low-rank regularization.

The fully sampled volume is an evaluation/development reference, not a target passed to the real GASSP1 trainer. If it is ever used to train a learned prior, evaluation must move to a held-out subject or acquisition to avoid anatomy leakage.

This phase distinguishes acquisition/subspace limitations from INR optimization limitations before real forward-model mismatch is introduced.

### Phase P3 — Real forward-model completion

Add B0/off-resonance phase evolution inside the non-Cartesian forward operator, preferably using time segmentation or a matched multifrequency formulation. Evaluate residual by true `|k|` radius rather than by readout sample index.

If structured outer residual remains after B0 modeling, audit in this order:

1. trajectory timing/gradient delay;
2. SMap consistency and coil noise whitening;
3. flow/cardiac phase and motion effects.

Strong outer-k weighting must not be selected before this audit, because it can force the reconstruction to fit samples where the physical model is least accurate.

### Phase P4 — Controlled high-frequency prior experiments

Use a one-variable-at-a-time sequence:

1. uniform noise-whitened complex L2 baseline;
2. mean-normalized/clipped DCF objective;
3. actual radial-shell-balanced objective;
4. with the selected objective fixed: no spatial prior versus joint wavelet versus coefficient LLR;
5. coefficient INR/multiscale capacity ablation.

Do not use the current `feng_rel` loss as the clean baseline. Do not use raw DCF as a backward-only preconditioner. Do not select a loss or regularizer from the final GIF alone.

Only if a hard physical basis remains the demonstrated bottleneck should a residual model be considered:

```text
X = Phi*C + R_high.
```

`R_high` must be spatially local/high-pass, low energy, data-consistent, and validated on held-out outer-k samples and a repeat scan. An unconstrained GAN, diffusion, or perceptual sharpening term is outside the present physics-grounded plan.

### Phase P5 — Real-data and repeatability validation

Run GASSP1 seed 0 first. Run seeds 1 and 2 only after the previous gates pass. Keep GASSP2 as a locked repeatability test rather than using it to tune the result.

For self-supervised model selection, hold out contiguous or blocked subsets of acquired outer-k samples, not isolated neighboring points. Use multiple fixed masks and report their variation.

## 8. Joint acceptance criteria

A final method must improve or preserve all four evidence groups:

1. **Measurement physics**
   - training and held-out complex k-space error;
   - actual radial-shell residual;
   - absence of a structured B0/trajectory-dependent residual.
2. **Image quality**
   - fixed-window visual anatomy;
   - magnitude, gradient, and high-pass error in retrospective data;
   - vessel-boundary edge spread and contrast;
   - no artificial rim or missing internal anatomy.
3. **T1 behavior**
   - shared-IJV curve and uncertainty;
   - matched B0/shared-mask MATLAB T1;
   - fit R2 and temporal second-difference behavior.
4. **Reproducibility**
   - seed stability;
   - GASSP1/GASSP2 repeatability of edges and T1;
   - no result that depends on one favorable hold-out mask.

A sharper image that worsens held-out k-space or fails to repeat is rejected. A plausible T1 obtained from a spatially incorrect reconstruction is also rejected.

## 9. Immediate work package

The first isolated coefficient-domain notebook has now been created; a reusable trainer remains future work:

```text
build_t1_basis.py
train_coeff_inr_spiral.py
notebooks/current/run_coeff_subspace_exact_gate.ipynb
results/coeff_subspace_exact_gate/<timestamp>/
```

The first execution batch should contain only:

1. P0 basis/projector/conditioning audit;
2. P1 exact K=5 coefficient synthetic recovery;
3. P2 retrospective coefficient-CG versus coefficient-INR.

No real-data seed sweep should start until these results have been reviewed.

## 10. Unresolved issues

1. The exact outer-k conditioning and recoverable spatial bandwidth of `A*Phi` have not been measured and saved.
2. B0/off-resonance is not present in the current training forward operator.
3. An inline direct coefficient-domain INR prototype exists in the unexecuted exact-gate notebook; a reusable trainer and real-data path do not yet exist.
4. K, Mz range, coefficient regularizer, and radial loss weighting have not passed retrospective selection.
5. Clean uniform-versus-DCF and radial-shell loss comparisons have not been completed under a common complex loss.
6. GASSP1/GASSP2 repeatability and three-seed stability are not established.
7. Strict matched B0/shared-mask MATLAB evaluation has not been completed for the latest target-free/subspace runs.
8. Some historical HL scripts point to a `Simulation` prior folder while the current three-parameter helper is stored in `Simulation_IR`; runtime helper resolution must be made explicit before claiming exact prior parity.

## 11. Literature basis for the proposed direction

The literature supports the components of this plan but does not guarantee success for this much shorter 50-frame, one-arm-per-TI acquisition.

1. Zhao B, et al. *Improved magnetic resonance fingerprinting reconstruction with low-rank and subspace modeling.* Magn Reson Med. 2018;79:933–942. [doi:10.1002/mrm.26701](https://doi.org/10.1002/mrm.26701).  
   Supports dictionary-derived temporal subspaces and direct reconstruction of spatial coefficient images.
2. Lingala SG, et al. *Accelerated dynamic MRI exploiting sparsity and low-rank structure: k-t SLR.* IEEE Trans Med Imaging. 2011;30:1042–1054. [doi:10.1109/TMI.2010.2100850](https://doi.org/10.1109/TMI.2010.2100850).  
   Supports combining low-rank structure with spatial sparsity rather than relying on a single temporal regularizer.
3. Lima da Cruz G, et al. *Sparsity and locally low rank regularization for MR fingerprinting.* Magn Reson Med. 2019;81:3530–3543. [doi:10.1002/mrm.27665](https://doi.org/10.1002/mrm.27665).  
   Supports dictionary compression plus sparsity/LLR for highly undersampled quantitative MRI.
4. Cao X, et al. *Optimized multi-axis spiral projection MR fingerprinting with subspace reconstruction for rapid whole-brain high-isotropic-resolution quantitative imaging.* Magn Reson Med. 2022;88:133–150. [doi:10.1002/mrm.29194](https://doi.org/10.1002/mrm.29194).  
   Supports the combined role of subspace reconstruction, LLR, and B0 correction in high-resolution spiral quantitative imaging.
5. Ostenson J, et al. *Multi-frequency interpolation in spiral magnetic resonance fingerprinting for correction of off-resonance blurring.* Magn Reson Imaging. 2017;41:63–72. [doi:10.1016/j.mri.2017.07.004](https://doi.org/10.1016/j.mri.2017.07.004).  
   Supports treating spiral off-resonance as a reconstruction-physics problem that affects image boundaries and quantitative maps.
6. Yaman B, et al. *Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data.* Magn Reson Med. 2020;84:3172–3191. [doi:10.1002/mrm.28378](https://doi.org/10.1002/mrm.28378).  
   Supports splitting acquired measurements into data-consistency and held-out loss sets.
7. Huang W, et al. *Neural Implicit k-Space for Binning-free Non-Cartesian Cardiac MR Imaging.* IPMI 2023. [doi:10.1007/978-3-031-34048-2_42](https://doi.org/10.1007/978-3-031-34048-2_42).  
   Demonstrates an INR approach to non-Cartesian k-space while explicitly acknowledging that small details not encoded in the acquired k-space may remain lost.

## 12. Current overall assessment

The project has passed the raw-layout/operator/T1 parity stage. It has also produced decisive negative evidence: unconstrained flexible INR can reach low DC with a wrong image, while a hard rank-4 physical prior can recover a plausible IJV curve without recovering acceptable anatomy.

The next scientifically justified step is therefore not another temporal-TV or epoch sweep. It is a direct coefficient-domain, protocol-aware reconstruction with an explicit synthetic/retrospective gate, a more complete spiral forward model, and local/multiscale spatial priors whose high-frequency contribution is tested against held-out acquired data and repeat scans.
