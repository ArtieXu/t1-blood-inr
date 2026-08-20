#!/usr/bin/env python3
"""
check_run.py -- Gate B acceptance check on a finished smoke-test log directory.

    python3 check_run.py runs/smoke_.../log/smoke32_5070_U0_260819_2210

Checks exactly the five things SETTING_PROGRESS_AND_TEST_PLAN_20260819.md sec.
Gate B asks for, and refuses to say more:

  loss finiteness          hard fail on any non-finite value in loss.csv
  loss trend               reported, warned on -- 32 steps cannot prove convergence
  seconds per step         reported from the cumulative time_s column
  peak VRAM                reported; warned near the 11.94 GiB device limit
  output shape/finiteness  recon_final.mat img_inr must be [50,1,216,216] and finite

Exit 0 = pass, 1 = fail, 2 = usage error.  A pass means the stack RUNS on this
GPU.  It is not a convergence, image-quality or scientific result.
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path

GASSP1_SHAPE = (50, 1, 216, 216)   # the real acquisition contract
VRAM_WARN_GIB = 10.5               # of ~11.94 GiB visible on the RTX 5070
TI_START_MS = 63                   # measured GASSP1 acquisition contract
PARITY_KB_GRID = 324               # what both scored notebooks hardcode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("--expect_epochs", type=int, default=32)
    args = ap.parse_args()

    d: Path = args.log_dir
    if not d.is_dir():
        print(f"not a directory: {d}", file=sys.stderr)
        return 2

    failures: list[str] = []
    warnings: list[str] = []
    expected_shape = GASSP1_SHAPE
    print(f"log directory: {d}")

    # ---------------------------------------------------------------- contract
    info_path = d / "run_info.json"
    if not info_path.is_file():
        failures.append("run_info.json missing")
        info = {}
    else:
        info = json.loads(info_path.read_text())
        print("\ncontract actually used")
        for key in ("script", "epochs", "seed", "frames", "coils", "grid_size",
                    "kb_grid_size", "dc_form", "dc_weighting_in_loss",
                    "dcf_in_backward", "dcf_norm", "time_coords",
                    "temporal_model", "data_path", "scale", "holdout_every"):
            print(f"  {key:24s} {info.get(key)}")
        # kb_grid_size None means torchkbnufft silently used 2*im_size = 432,
        # while both scored notebooks hardcode 324.  Only a ~9e-4 operator
        # difference, but it should never be left unrecorded.
        if info.get("kb_grid_size") is None:
            warnings.append("kb_grid_size is null -> torchkbnufft used its own "
                            f"default 2*{info.get('grid_size')} instead of the "
                            f"{PARITY_KB_GRID} both notebooks use; pin --kb_grid_size")
        elif info.get("kb_grid_size") != PARITY_KB_GRID:
            warnings.append(f"kb_grid_size={info.get('kb_grid_size')} differs from the "
                            f"{PARITY_KB_GRID} both notebooks use")
        ti = info.get("ti_ms") or []
        if ti:
            print(f"  {'ti_ms':24s} {ti[0]} .. {ti[-1]}  (contract: {TI_START_MS} + 200*n)")
            if ti[0] != TI_START_MS:
                warnings.append(f"ti_ms starts at {ti[0]}, expected {TI_START_MS} "
                                "(measured-acquisition contract)")
        if info.get("epochs") != args.expect_epochs:
            failures.append(f"epochs={info.get('epochs')} != expected {args.expect_epochs}")
        if info.get("cg_supervision") is not False or info.get("target_free") is not True:
            failures.append("run_info says this was not the target-free entrypoint")
        # trust the run's own contract for the shape check, then separately flag
        # any deviation from the real GASSP1 acquisition
        if info.get("frames") and info.get("grid_size"):
            expected_shape = (int(info["frames"]), 1,
                              int(info["grid_size"]), int(info["grid_size"]))
            if expected_shape != GASSP1_SHAPE:
                warnings.append(f"run contract {expected_shape} is not the GASSP1 "
                                f"acquisition {GASSP1_SHAPE} -- synthetic or reduced data?")

    # -------------------------------------------------------------- loss.csv
    loss_path = d / "loss.csv"
    if not loss_path.is_file():
        failures.append("loss.csv missing")
    else:
        with loss_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        print(f"\nloss.csv: {len(rows)} rows")
        if len(rows) != args.expect_epochs:
            failures.append(f"loss.csv has {len(rows)} rows, expected {args.expect_epochs}")

        holdout_on = bool(info.get("holdout_every"))
        nonfinite: list[str] = []
        for row in rows:
            for key, value in row.items():
                if value in ("", None):
                    continue
                try:
                    v = float(value)
                except ValueError:
                    continue
                # holdout columns are nan BY DESIGN only when --holdout_every is 0.
                # If holdout is enabled they must carry real numbers; a nan there
                # means the frame split never took effect.
                if math.isnan(v) and key.startswith("dc_holdout"):
                    if not holdout_on:
                        continue
                    nonfinite.append(f"epoch {row['epoch']} {key}=nan "
                                     "(holdout enabled but column is nan)")
                    continue
                if not math.isfinite(v):
                    nonfinite.append(f"epoch {row['epoch']} {key}={value}")
        if nonfinite:
            failures.append(f"non-finite values in loss.csv: {nonfinite[:6]}"
                            f"{' ...' if len(nonfinite) > 6 else ''}")
        else:
            print("  all values finite"
                  + ("" if holdout_on else
                     " (holdout columns are nan by design: --holdout_every 0)"))
            if holdout_on:
                last = rows[-1]
                print(f"  holdout split active: "
                      f"train_uniform_rel={float(last['dc_train_uniform_rel']):.4e}  "
                      f"holdout_uniform_rel={float(last['dc_holdout_uniform_rel']):.4e}")

        if rows:
            def col(name): return [float(r[name]) for r in rows]
            n = max(1, len(rows) // 4)
            dc_first, dc_last = col("dc")[:n], col("dc")[-n:]
            m0, m1 = sum(dc_first) / n, sum(dc_last) / n
            print(f"\ntrend (first {n} vs last {n} steps)")
            print(f"  dc            {m0:.6e} -> {m1:.6e}   "
                  f"({'down' if m1 < m0 else 'UP'} {abs(m1 - m0) / max(m0, 1e-30) * 100:.1f}%)")
            print(f"  dc_uniform_rel {col('dc_uniform_rel')[0]:.6e} -> "
                  f"{col('dc_uniform_rel')[-1]:.6e}")
            print(f"  total         {col('total')[0]:.6e} -> {col('total')[-1]:.6e}")
            print(f"  eps_frac      {col('eps_frac')[0]:.4f} -> {col('eps_frac')[-1]:.4f}"
                  "   (fraction of predicted samples under the relative-L2 epsilon)")
            if m1 >= m0:
                warnings.append(f"dc did not decrease over {len(rows)} steps; check "
                                "the learning rate before spending a longer run")

            times = col("time_s")                       # cumulative seconds
            per_step = [times[0]] + [b - a for a, b in zip(times, times[1:])]
            steady = per_step[1:] or per_step           # step 1 includes warm-up
            print(f"\ntiming")
            print(f"  step 1 (incl. CUDA/tcnn warm-up)  {per_step[0]:.3f} s")
            print(f"  steady-state mean                 {sum(steady)/len(steady):.3f} s/step")
            print(f"  steady-state max                  {max(steady):.3f} s/step")
            print(f"  total train time                  {times[-1]:.1f} s")
            print(f"  extrapolated 1600 steps           "
                  f"{sum(steady)/len(steady) * 1600 / 60:.1f} min "
                  "(schedule differs; timing only)")

            peak = max(col("gpu_peak_gb"))
            print(f"\nmemory")
            print(f"  peak torch-allocated VRAM         {peak:.3f} GiB")
            print("  NOTE tiny-cuda-nn allocates outside the torch caching "
                  "allocator; compare nvidia_smi_after.txt for the true device total.")
            if peak > VRAM_WARN_GIB:
                warnings.append(f"peak VRAM {peak:.2f} GiB is close to the "
                                f"{VRAM_WARN_GIB} GiB warning line")

    # ------------------------------------------------------------- output data
    recon = d / "recon_final.mat"
    if not recon.is_file():
        failures.append("recon_final.mat missing")
    else:
        try:
            import numpy as np
            from scipy import io
            mat = io.loadmat(recon)
            img = mat["img_inr"]
            print(f"\nrecon_final.mat")
            print(f"  img_inr shape                     {img.shape} {img.dtype}")
            print(f"  finite                            {bool(np.isfinite(img).all())}")
            print(f"  max |img_inr|                     {float(np.abs(img).max()):.6e}")
            print(f"  scale (physical units factor)     {float(mat['scale'].ravel()[0]):.6e}")
            if tuple(img.shape) != expected_shape:
                failures.append(f"img_inr shape {img.shape} != {expected_shape}")
            if not np.isfinite(img).all():
                failures.append("img_inr contains non-finite values")
            if float(np.abs(img).max()) == 0.0:
                failures.append("img_inr is identically zero")
        except Exception as exc:                                  # noqa: BLE001
            failures.append(f"could not read recon_final.mat: {type(exc).__name__}: {exc}")

    for name in ("final_state.pt", "final_residual_by_frame.csv", "recon_final_abs.png"):
        p = d / name
        print(f"  {'OK  ' if p.is_file() else 'MISS'} {name}"
              f"{f'  ({p.stat().st_size} bytes)' if p.is_file() else ''}")
        if not p.is_file():
            failures.append(f"{name} missing")

    # ------------------------------------------------------------------ result
    print("\n" + "=" * 68)
    for w in warnings:
        print(f"WARNING: {w}")
    if failures:
        print("GATE B FAILED:")
        for f in failures:
            print(f"- {f}")
        return 1
    print("GATE B PASSED -- the 5070 stack runs, outputs are finite and correctly shaped.")
    print("This is a RUNTIME result only. It is not convergence, not image quality,")
    print("and not a scientific result. The stop gate RANK5_MODEL_GAP_DETECTED is")
    print("unchanged: no measured-data tuning until a revised prior passes an")
    print("independent image gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
