# T1_blood_INR

INR reconstruction of GASSP1 single-shot golden-angle spiral data (IJV T1 blood).

> Production INR now uses the 50 physical TI frames. The validated historical
> 59-frame path (`gassp1_data_59.mat`, `parity_step_*`) is retained only for
> CG/TV parity diagnostics. See `PROJECT_INDEX.md` and `HANDOVER.md`.

## Pipeline

1. **MATLAB export** — `export_gassp1.m` reads the Philips raw and writes `gassp1_data.mat`.
2. **Python adapter** — `load_spiral.py` loads `gassp1_data.mat`, builds trajectory/k-space/maps. Run it directly for the adjoint validation gate (`adjoint_check.png`). *Done: gate passed; density (wi) weighting confirmed necessary.*
3. **Current INR recon** — `train_inr_from_cg.py` runs a matched 50-frame
   fixed-CG control and a candidate whose CG anchor decays to zero.
4. **Primary evaluation** — fully sampled Spiral supplies the shared IJV mask
   and reference T1. `evaluate_inr50_candidates.m` applies matched B0 deblur
   and the same MATLAB T1 fitter before the final comparison.

## Run the recon (GPU required)

Needs torch + tiny-cuda-nn + torchkbnufft + CUDA. This folder is **self-contained** (`model.py` and `utils.py` are copied in) so it can be uploaded and run anywhere.

### Colab (recommended)
Open `notebooks/current/run_inr50_fullref_candidates.ipynb` from the organized
Drive `01_current` folder, select a GPU runtime, and run top to bottom.

### Command line
```bash
python3 train_inr_from_cg.py -g 0 \
  --data_path gassp1_data.mat --cg_path cg_predeblur.mat \
  --kb_grid_size 324 --no_support_mask --loss_norm relative \
  --pretrain_epochs 300 --finetune_epochs 50 \
  --cg_weight_start 1 --cg_weight_end 0 -t 0 -st 0 -l 0 --dcf_norm none
```

Outputs are written under `log/<tag>_<timestamp>/`. CG agreement is a handoff
diagnostic, not ground truth. Final selection uses the fully sampled reference
after matched B0 deblur and a shared-mask T1 fit.

### Stage C files
- `spiral_nufft.py` — `SpiralNUFFT`: torchkbnufft operator, explicit `grid_size=216`, 1 arm/frame, 3315 samples. Replaces the radial `utils.NUFFT`.
- `inr_spiral.py` — `SpiralINR(INR)`: wi-weighted data-consistency loss + visual-only `infer`. All else inherited.
- `main_spiral.py` — driver: load data → spiral NUFFT → INR fit.

The original repo (`main.py`, `model.py`, `utils.py`) is untouched.

## Data

- **Recon target:** `.../20240926_ML/Raw/20240926_170304_IJVT1_GASSP1_Auto.raw` — 1 spiral arm per dynamic phase, golden angle (137.51°) between phases.
- **Reference only:** `.../20240926_ML/Raw/20240926_170055_IJVT1_Spiral.raw` — fully-sampled spiral, kept as an independent reference for result comparison. Not reconstructed by INR.

## `gassp1_data.mat` contract (output of `export_gassp1.m`, `-v7.3`)

| var | shape | meaning |
|-----|-------|---------|
| `k_data` | `[3315, 50, C]` complex | measured k-space (samples, phases, coils) |
| `Traj`   | `[3315, 50]` complex | GA-rotated trajectory, normalized ~±0.5 |
| `S`      | `[216, 216, C]` complex | coil sensitivity maps (masked) |
| `Sinv`   | `[216, 216, C]` complex | inverse sens maps (masked) |
| `mask`   | `[216, 216]` | support mask |
| `N`      | scalar | grid size = 216 (180 × 1.2 OS) |
| `shift`  | `[2]` | FOV-shift for k-space phase ramp |
| `dwell`  | scalar | sample dwell time (µs) |

`C` = coil count (from raw). Density compensation `wi` is computed Python-side (port of `calc_dcf_Spiral`); not exported here.
