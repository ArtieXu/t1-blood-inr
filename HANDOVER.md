# Handover: T1 Blood INR / MATLAB-Python Spiral CG Parity

Last updated: 2026-07-11

Workspace root: `/ibic/projects3/ORCA/zechenxu`

Primary project: `INR_for_DynamicMRI/T1_blood_INR`

MATLAB reference: `JHU/T1_Blood/20240926_ML/Recon_0926_ML.m`

## Task

We are diagnosing why the Python GASSP1 reconstruction could not reproduce the
MATLAB CG reconstruction and produced objectionable artifacts and intensity
behavior. The immediate scientific endpoint is reliable IJV blood T1 fitting,
not merely obtaining a low data-consistency loss.

The work progressed through these gates:

1. Match MATLAB raw-data layout, trajectory, DCF, S/Sinv, shift, and I0.
2. Reproduce the MATLAB Sinv-based nonlinear CG update in Python.
3. Recover the historical 59-frame reconstruction protocol.
4. Compare multiple CG iterations rather than only final DC.
5. Apply the same historical B0 deblur to MATLAB and Python.
6. Run the same vessel segmentation and weighted T1 fit.

## Current Status

There is no active process or unfinished job. The requested 59-frame parity,
B0 deblur, and downstream T1 fitting are complete.

The corrected Python parity reconstruction now reproduces the clinically
important IJV intensity curve and T1 result:

| Case | T1 (ms) | R2 |
| --- | ---: | ---: |
| Fully sampled spiral, B0 deblur | 1880.831 | 0.995674 |
| MATLAB CG, pre-deblur | 2025.778 | 0.989709 |
| Python parity CG 59->50, pre-deblur | 2030.546 | 0.989910 |
| MATLAB CG, B0 deblur | 2004.480 | 0.991344 |
| Python parity CG 59->50, B0 deblur | 2008.671 | 0.991524 |

Python and MATLAB differ by only 4.77 ms before deblur and 4.19 ms after
deblur. The original Python pixel-TV-CG result was 1453.956 ms with R2 0.718,
so the corrected protocol has resolved the important intensity/T1 failure.

## Main Findings

### INR production update after the parity handover

The 59-frame extension is required by the historical CG temporal-TV boundary
condition, not by the INR coordinate representation. The current production
experiment therefore uses the 50 physical TI frames and keeps signed-59 only as
an operator/CG diagnostic.

The next controlled run is
`notebooks/current/run_inr50_fullref_candidates.ipynb`:

1. 50-frame fixed-CG control;
2. identical 50-frame run with CG weight decaying from 1 to 0;
3. fully sampled shared-mask screening in Colab;
4. strict matched B0-deblur and T1 comparison with
   `evaluate_inr50_candidates.m` on ORCA.

CG agreement is now only a parity/handoff gate. The fully sampled Spiral result
is the primary scientific reference. This update supersedes the older suggestion
below to make signed-59 the default production INR input; signed-59 remains
available for ablation and provenance.

### 1. The historical protocol is 59 frames, not 50

The raw acquisition contains 55 phases. The historical MATLAB reconstruction
formed 59 frames by prepending the last four phases:

- Trajectory: `[last 4, all 55]`
- K-space: `[-last 4, all 55]` (the minus sign is essential)
- DCF: computed after extension, using all 59 arms
- Initial image: sum frames 5:54, then repeat that image over 59 frames
- CG: three separate 15-iteration calls, with optimizer state reset per call
- Final output: crop frames 5:54 to obtain the 50 TI images

The currently active lines in `Recon_0926_ML.m` instead crop to 50 before CG.
That is not the protocol that produced `new_initial.mat`.

Computing DCF after pre-cropping to 50 changes I0 by exactly `59/50 = 1.18`.
After removing that scale, the current 50-frame I0 and historical middle 50
frames agree to approximately `2e-14`.

### 2. The Python forward operator is not the primary problem

With matched trajectory, DCF, shift, maps, and Sinv pseudo-adjoint:

- 59-frame Python I0 versus MATLAB I0: approximately 0.031% error
- First CG update versus MATLAB: approximately 0.008% error
- DC at matched checkpoints agrees at approximately `1e-5` relative error
- Framewise intensity remains stable

The poor original Python image came mainly from using a different optimization
problem and gradient, not from corrupt raw data.

### 3. MATLAB CG does not use the Hermitian S-map adjoint

MATLAB uses `Sinv` in its backward operator. Standard PyTorch autograd through
the forward model uses `conj(S)`, which is a different gradient. The parity
solver therefore computes the DC gradient explicitly with the Sinv
pseudo-adjoint.

MATLAB also uses an unnormalized sum objective:

`0.5 * sum(w * |E(x)-k|^2) + lambda * sum(|Dt*x|)`

For the 59-frame case:

- `lambda = 0.1 * max(abs(I0)) = 1.153154...`
- Do not substitute the old Python mean loss with `temporal_tv_weight=0.1`.
- Do not compare raw DC numbers from differently normalized objectives.

### 4. Nonlinear-CG trajectories still diverge numerically

The Python and current MATLAB 59-frame checkpoints have these complex-image
relative errors:

| Iteration | Relative error |
| ---: | ---: |
| 1 | 0.00795% |
| 5 | 0.0942% |
| 15 | 0.635% |
| 30 | 4.17% |
| 45 | 9.50% |

This is expected from small NUFFT differences accumulating through nonlinear
conjugate directions when the backward operator is a non-Hermitian Sinv
pseudo-adjoint. Despite this, the framewise magnitude energy is stable and the
downstream T1 agrees with MATLAB.

The current MATLAB 59-frame rerun itself differs from the historical saved
`I_GASSP_V_a1` by 5.47%, even though I0 and initialization match almost exactly.
This indicates historical software/numerical solver provenance remains, not a
remaining gross Python data-layout error.

### 5. B0 deblur is now matched

Python parity CG was processed using the exact JHU `.p` implementation:

`Spiral_Deblur(I, B0, Traj, dwell, 15, 1)`

Python versus historical MATLAB after B0 deblur:

- Complex relative error: 24.70%
- Magnitude relative error: 9.95%
- Frame energy ratio mean: 0.995742
- Frame energy ratio standard deviation: 0.000928
- Frame energy ratio range: 0.992770 to 0.999989

The large complex error is phase sensitive. It does not translate into a T1
failure: B0-deblurred Python and MATLAB T1 differ by only 4.19 ms.

## Files Added or Changed

### Source/diagnostic code

- `INR_for_DynamicMRI/T1_blood_INR/export_gassp1_59.m`
  - Exports the historical signed 59-frame dataset.
  - Writes only `gassp1_data_59.mat`; it does not replace `gassp1_data.mat`.

- `INR_for_DynamicMRI/T1_blood_INR/parity_step_matlab.m`
  - Supports `PARITY_FRAMES=50` or `59`.
  - Runs the original MATLAB CG helper and exports iteration checkpoints.
  - In 59 mode, checks I0 and final cropped output against `new_initial.mat`.

- `INR_for_DynamicMRI/T1_blood_INR/parity_step_python.py`
  - Supports `--data`, `--matlab_ref`, and `--tag`.
  - Implements explicit Sinv-CG with MATLAB TV, Armijo line search, and three
    optimizer blocks.
  - Produces checkpoint JSON, final MAT, and montage.

- `INR_for_DynamicMRI/T1_blood_INR/deblur_parity_cg_59.m`
  - Applies the historical JHU B0 deblur to corrected Python CG.
  - Asserts that MATLAB resolves the JHU `.p` helper.

- `INR_for_DynamicMRI/T1_blood_INR/fit_ijv_t1_matlab_python.m`
  - Existing default four-case behavior remains available.
  - Optional environment variables add parity/B0 cases without overwriting old
    outputs: `T1_PARITY_FILE`, `T1_PARITY_VAR`, `T1_MATLAB_CG_VAR`.

### Important generated artifacts

- `INR_for_DynamicMRI/T1_blood_INR/gassp1_data_59.mat`
- `INR_for_DynamicMRI/T1_blood_INR/parity_matlab_ref_59.mat`
- `INR_for_DynamicMRI/T1_blood_INR/parity_python_report_59.json`
- `INR_for_DynamicMRI/T1_blood_INR/parity_python_cg_59.mat`
- `INR_for_DynamicMRI/T1_blood_INR/parity_multi_compare_59.png`
- `INR_for_DynamicMRI/T1_blood_INR/parity_python_cg_59_deblur.mat`
- `INR_for_DynamicMRI/T1_blood_INR/t1_ijv_compare/ijv_t1_summary_parity59.csv`
- `INR_for_DynamicMRI/T1_blood_INR/t1_ijv_compare/ijv_t1_summary_b0deblur_parity59.csv`
- `INR_for_DynamicMRI/T1_blood_INR/t1_ijv_compare/ijv_t1_matlab_python_b0deblur_parity59.png`
- `INR_for_DynamicMRI/T1_blood_INR/t1_ijv_compare/dynamic_montage_b0deblur_parity59.png`

The latest 59-frame parity/B0 files were produced locally. They were not
uploaded to Google Drive during this final phase; Drive copies may be stale.

## Reproduction Commands

Run from `/ibic/projects3/ORCA/zechenxu`.

### Python environment

The default Python does not have the required Torch stack. Use FreeSurfer
Python. `/tmp/torchkbnufft_cpu` is temporary and may disappear between
sessions. Recreate it when needed:

```bash
/usr/local/freesurfer/8.1.0/python/bin/python3 -m pip install \
  --no-deps --target /tmp/torchkbnufft_cpu \
  /ibic/projects3/ORCA/zechenxu/torchkbnufft-1.4.0-py3-none-any.whl
```

### Export historical 59-frame data

The MAT already exists; rerun only if the raw export must be regenerated.

```bash
/usr/local/MATLAB/R2023a/bin/matlab -batch \
  "run('/ibic/projects3/ORCA/zechenxu/INR_for_DynamicMRI/T1_blood_INR/export_gassp1_59.m')"
```

### Generate MATLAB 59-frame checkpoints

```bash
PARITY_FRAMES=59 /usr/local/MATLAB/R2023a/bin/matlab -batch \
  "run('/ibic/projects3/ORCA/zechenxu/INR_for_DynamicMRI/T1_blood_INR/parity_step_matlab.m')"
```

### Run Python 59-frame parity CG

```bash
PYTHONPATH=/tmp/torchkbnufft_cpu \
MPLCONFIGDIR=/tmp/matplotlib-cache \
/usr/local/freesurfer/8.1.0/python/bin/python3 \
  INR_for_DynamicMRI/T1_blood_INR/parity_step_python.py \
  --device cpu --threads 4 \
  --matlab_ref parity_matlab_ref_59.mat \
  --data gassp1_data_59.mat --tag _59
```

### Apply historical B0 deblur

```bash
/usr/local/MATLAB/R2023a/bin/matlab -batch \
  "run('/ibic/projects3/ORCA/zechenxu/INR_for_DynamicMRI/T1_blood_INR/deblur_parity_cg_59.m')"
```

### Run matched B0-deblurred T1 comparison

```bash
T1_PARITY_FILE=/ibic/projects3/ORCA/zechenxu/INR_for_DynamicMRI/T1_blood_INR/parity_python_cg_59_deblur.mat \
T1_PARITY_VAR=I_python_parity_deblur \
T1_MATLAB_CG_VAR=I_deblur_unwrap_1a \
/usr/local/MATLAB/R2023a/bin/matlab -batch \
  "run('/ibic/projects3/ORCA/zechenxu/INR_for_DynamicMRI/T1_blood_INR/fit_ijv_t1_matlab_python.m')"
```

MATLAB commands may need to run outside a restricted network sandbox so they
can reach `mlcl2026.lic.uw.edu:27000`. Use
`/usr/local/MATLAB/R2023a/bin/matlab`; `/usr/local/bin/matlab` may select an
R2025b installation that requests MathWorks sign-in.

## Where Work Is Still Incomplete

1. Exact final complex-image parity is not achieved. Python differs from the
   current MATLAB 59-frame iter-45 result by 9.50% and from the historical
   saved result by 10.04%. T1 parity is already achieved, so only pursue exact
   complex parity if it is scientifically required.

2. B0 deblur reduces the CG T1 by about 21-22 ms, but MATLAB/Python CG remain
   about 124-128 ms above the fully sampled reference. The cause of this
   remaining gap has not been isolated.

3. The current fitter independently runs `Vessel_Segment` for every method.
   Different masks can contribute to differences. A high-value next test is
   to define one shared IJV mask and fit every reconstruction with it.

4. The corrected parity solver has not been merged into
   `pixel_tv_cg_recon.py`, `main_spiral.py`, or the INR training path. Those
   older entrypoints must not be described as corrected merely because the
   parity script now works.

5. The INR model has not been rerun with the corrected historical 59-frame
   protocol. Do not add low-rank regularization until the corrected direct
   pixel/CG baseline is explicitly used by the production path.

## Recommended Next Steps

1. Add a shared-mask T1 comparison using the MATLAB CG B0-deblur mask or the
   fully sampled mask. Quantify how much of the remaining 124-128 ms gap comes
   from segmentation versus the ROI signal curve.

2. Compare the fully sampled and CG ROI curves frame by frame under that fixed
   mask. Check TI ordering, inversion null location, and late-TI plateau.

3. Decide the goal before changing the solver:
   - If the goal is accurate blood T1, the Python parity baseline is already
     validated within approximately 4 ms of MATLAB.
   - If the goal is exact complex-image parity, instrument MATLAB per-iteration
     step size, line-search count, beta, gradient norm, and objective, then
     compare those values with `parity_step_python.py`.

4. For INR, use the corrected 59-frame pre-deblur CG as the operator-consistent
   baseline or warm start. Apply B0 as the same post-process, unless B0 is first
   added to the INR forward model. Do not warm-start an off-resonance-free
   forward model from a deblurred target without acknowledging that mismatch.

5. Only after the corrected 59-frame direct-pixel/CG path is integrated should
   low-rank or additional INR regularizers be tested.

## Pitfalls That Must Not Be Repeated

- Do not pre-crop the historical dataset to 50 frames before DCF/CG.
- Do not forget the negative sign on the four prepended k-space frames.
- Do not apply the negative sign to the prepended trajectory.
- Do not compute 50-arm DCF and treat it as 59-arm DCF; this causes the exact
  1.18 I0 scale discrepancy.
- Do not use `conj(S)` autograd and call it the MATLAB gradient. MATLAB uses
  the Sinv pseudo-adjoint.
- Do not compare differently normalized DC values as if they are the same
  objective.
- Do not use Python `temporal_tv_weight=0.1` as a substitute for MATLAB's
  unnormalized TV objective.
- Do not run one continuous 45-iteration NCG and call it equivalent to three
  separate 15-iteration MATLAB calls; each MATLAB call resets CG state.
- Do not assume every block reaches 15 updates. The 50-frame diagnostic ran
  `15+15+2` because the third block met tolerance; the historical 59-frame
  reconstruction ran the full `15+15+15`.
- Do not infer image quality from a lower DC alone. The earlier INR reached
  lower DC but had poor intensity and T1 behavior.
- Do not use `DISC_Scan/20251020_RA/func_old/Spiral_Deblur.m` for this step.
  It has a different five-argument signature. Use the JHU
  `20240926_ML/functions/Spiral_Deblur.p` six-argument implementation.
- Do not ignore HDF5/MATLAB axis reversal. `load_spiral.py` transposes S, Sinv,
  and mask spatial axes for a reason.
- Do not simultaneously apply a shift phase ramp and NUFFT `n_shift`. The
  validated parity path uses `apply_shift=False` plus the matching `n_shift`.
- Do not introduce an extra support mask in parity. S and Sinv are already
  masked; the validated parity operator uses `support_mask=None`.
- Do not assume `new_initial.mat` was generated by the currently active
  50-frame lines in `Recon_0926_ML.m`; it contains the historical 59-frame
  reconstruction state.
- Do not overwrite the old four-case fitting outputs. The fitter's environment
  variables intentionally create tagged parity/B0 outputs.
- Do not trust a MATLAB run that silently resolved a different helper. Check
  `which` for duplicate MATLAB functions, especially `Spiral_Deblur`,
  `CG_Nonlinear_wNorms`, and `reg_fun`.
- Do not switch to `/usr/local/bin/matlab` when R2023a works; that symlink may
  trigger an R2025b sign-in prompt.
- Do not assume `/tmp/torchkbnufft_cpu` persists into the next session.

## Key Interpretation for the Next Session

The central operator/protocol bug is resolved. Python can reproduce the
MATLAB I0, first CG updates, framewise intensity, and downstream IJV T1. The
remaining pixelwise complex discrepancy is an optimization/numerical-path
issue, while the remaining fully sampled-versus-CG T1 difference is now the
scientifically relevant open question.
