# T1 blood INR project index

## Current status update (2026-08-11)

The current evidence, diagnosis, protocol-aware coefficient-domain direction,
and gated execution plan are recorded in
[`PROGRESS_AND_OUTLOOK_20260811.md`](PROGRESS_AND_OUTLOOK_20260811.md).

This update supersedes the older production workflow below as the next research
direction. Preserve the older section for provenance; do not run its notebook as
the default next experiment.

## Historical workflow (retained for provenance)

1. Run `notebooks/current/run_inr50_fullref_candidates.ipynb` on a Colab GPU.
2. Compare the fixed-CG control and CG-release candidate inside the notebook.
3. Sync their two `recon_final.mat` files back to ORCA.
4. Run `evaluate_inr50_candidates.m` for matched B0 deblur, one shared IJV
   mask, and the same MATLAB T1 fitter.
5. If CG release improves DC but introduces temporal oscillation, test a weak
   second-difference penalty next. Do not add first-difference TV or low-rank
   regularization before that failure is observed.

## Historical directory snapshot and diagnostic files

| Role | Files | Status |
|---|---|---|
| Current research plan | `PROGRESS_AND_OUTLOOK_20260811.md` | Follow the gated coefficient-domain flow |
| First validation notebook | `notebooks/current/run_coeff_subspace_exact_gate.ipynb` | Implemented, unexecuted; run exact synthetic gate first |
| Production input | `gassp1_data.mat`, `cg_predeblur.mat` | 50 physical TI frames |
| Historical candidate notebook | `notebooks/current/run_inr50_fullref_candidates.ipynb` | Superseded as the default next run |
| Primary reference | `results/full_spiral_reference.mat`, `results/full_spiral_shared_mask.mat` | Fully sampled, B0 deblurred |
| Strict evaluator | `evaluate_inr50_candidates.m` | Run after the two INR outputs exist |
| Operator parity | `parity_step_python.py`, `parity_step_matlab.m` | Passed forward/early-update gates |
| Historical TV context | `gassp1_data_59.mat`, `cg_predeblur_59.mat` | Diagnostic only; 59 then crop to 50 |
| Trial notebooks | `notebooks/validation/` | Executed evidence, not production entrypoints |
| Superseded entrypoints | `notebooks/archive/` | Kept as records |

## Directory rules

- Root: active source, MATLAB/Python workflow scripts, and small index files.
- `notebooks/current/`: notebook to run now.
- `notebooks/validation/`: parity, normalized-loss, signed-59, and pixel-CG trials.
- `notebooks/archive/`: superseded general reconstruction notebooks.
- `results/`: reconstruction/reference results.
- `t1_ijv_compare/`: earlier T1 comparisons and figures.
- `backup/cg_debug_20260711/`: frozen early parity investigation.

Nothing in the cleanup was deleted. The 59-frame artifacts and prior notebooks
remain available for provenance, but they should not be used as the default INR
production path.

Drive note: the legacy `gassp1_data.mat` remains at the v2 root because that
individual upload did not grant the connected Drive app write access. The
current notebook checks `02_data_reference/` first and then falls back to the
root file, so no duplicate 95 MB copy is needed.

The full fully sampled image remains on ORCA. Drive receives only
`full_spiral_shared_curve.mat` (shared mask, 50-point ROI curve, and fit
parameters), which is sufficient for Colab screening without uploading the
full research image volume.
