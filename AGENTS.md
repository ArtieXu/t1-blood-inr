# T1 Blood INR project instructions

## Read first

Before changing code or launching a run, read these files in order:

1. `MIGRATION_5070.md`
2. `MIGRATION_5070_MANIFEST.tsv`
3. `PROJECT_INDEX.md`
4. `PROGRESS_AND_OUTLOOK_20260811.md`

`MIGRATION_5070.md` contains the newest status. The older handover and project
index remain useful provenance, but their status sections are not current.

## Scientific objective

The endpoint is reliable IJV blood T1 estimation and image fidelity, not merely
a small k-space data-consistency loss.

Report these evidence levels separately:

- source/static validation;
- completed execution and checkpoint provenance;
- objective convergence;
- image-domain gates;
- scientific interpretation and its limits.

A sharp or data-consistent CG image proves feasibility under the selected
forward model and prior; it does not prove unique recovery. A low DC loss alone
does not prove a correct reconstruction.

## Current stop gate

The verified P2 retrospective decision is `RANK5_MODEL_GAP_DETECTED`. The hard
rank-5 physical T1 subspace fails the oracle image gate, so do not start
measured-data tuning until a revised prior passes an independent gate.

Preserve the main T1 component when testing a soft constraint or a small
residual dynamic subspace. Do not describe the P2 result as disproving the T1
model.

## Experiment contracts

- Use reference TI `50 + 200*n` ms for the retrospective reference experiment.
- Use acquisition TI `63 + 200*n` ms only for the current measured acquisition.
- A strict ablation keeps data, trajectory, DCF, seed, training steps,
  checkpoint selection, scoring, software, and hardware matched.
- C0/C1 compare uniform versus DCF rank-5 objectives; they are not a no-T1
  control. `F0_noT1_DCF` is the matched flexible no-T1 arm.
- Do not mix a Tesla T4 arm with an RTX 5070 arm and call it a strict
  one-variable comparison. Rerun all compared arms on the same 5070 stack.
- Preserve old T4 notebooks and results. Port them through a separate 5070
  entrypoint and record the complete runtime signature.
- Select checkpoints using the arm's own k-space objective, never the reference
  image score.

## 5070 workflow

1. Run `python3 verify_migration.py` to validate required local files.
2. Build a Blackwell-compatible environment and run
   `python3 verify_migration.py --env`.
3. Run a 32-step smoke test and record loss, seconds/step, peak VRAM, and output
   finiteness.
4. Run a matched short parity experiment before any full 800/1600-step study.
5. Keep 5070 outputs in a new result directory; never overwrite T4 evidence.

## Data and security

- Treat paths in `MIGRATION_5070_MANIFEST.tsv` as relative to this directory.
- Verify size and SHA-256 before using copied local data.
- A Drive URL is provenance, not proof that the file is accessible. Verify the
  account, file ID, metadata, and downloaded checksum.
- Never print, copy, commit, or upload API keys, W&B keys, tokens, credentials,
  or patient-identifying data.
- `notebooks/current/run_inr_unsup_spiral_v2.ipynb` contains a credential-like
  W&B token string. Revoke/rotate it and sanitize the notebook before any
  full-folder transfer.

## Communication

- Explain each new acronym or technical term briefly in Chinese on first use.
- For status questions, give completed gates, current blocker, and next gate.
- Keep claims bounded by the exact artifact and execution evidence inspected.
