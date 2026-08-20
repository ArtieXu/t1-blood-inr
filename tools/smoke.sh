#!/usr/bin/env bash
# smoke.sh -- Gate B: one 32-step RTX 5070 smoke test.
#
#     source tools/env.sh
#     bash   tools/smoke.sh          # arm U0 (script defaults)
#     ARM=U1 bash tools/smoke.sh     # highest backward cost
#
# WHAT THIS IS: a runtime check -- does the stack run, how fast, how much VRAM,
# are the outputs finite and the right shape.
# WHAT THIS IS NOT: a scientific result, a convergence result, or a prefix of a
# 1600-step run.  train_inr_unsup_spiral.py derives both the coarse-to-fine
# schedule (epochs_per_level = epochs // n_levels) and the StepLR step size
# (epochs // 2) FROM --epochs, so a 32-step run follows a different LR/level
# schedule than a 1600-step run.  Per-step time and peak VRAM transfer; the loss
# trajectory does not.
#
# Outputs land under runs/ (override with T1_OUT_ROOT).  Nothing is written
# to any existing T4 result directory.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARM="${ARM:-U0}"
EPOCHS="${EPOCHS:-32}"
SEED="${SEED:-0}"
# train_inr_unsup_spiral.py's --kb_grid_size defaults to None, which makes
# torchkbnufft pick its own default of 2*im_size = 432.  BOTH scored notebooks
# (build_p2_..._notebook.py:278 and run_coeff_subspace_exact_gate.ipynb cell 8)
# hardcode 324, and the trainer's own help string calls 324 "parity-validated".
# Leaving this unset therefore runs a DIFFERENT forward operator than every
# scored experiment, and costs ~1.7 GiB more VRAM at 28 coils.  Pin it.
KB_GRID="${KB_GRID:-324}"
OUT_ROOT="${T1_OUT_ROOT:-${ROOT}/runs}"
OUT="${OUT_ROOT}/smoke_${ARM}_${STAMP}"

# The three DCF-placement arms documented in the header of
# train_inr_unsup_spiral.py.  U0 is the script's own default.
case "${ARM}" in
    U0) ARM_ARGS=(--dc_form feng_rel   --dc_weighting uniform) ;;
    U1) ARM_ARGS=(--dc_form feng_rel   --dc_weighting uniform --dcf_backward) ;;
    U2) ARM_ARGS=(--dc_form global_rel --dc_weighting dcf) ;;
    *)  echo "ARM must be U0, U1 or U2 (got '${ARM}')" >&2; exit 2 ;;
esac

if [ -z "${VIRTUAL_ENV:-}" ] || [ -z "${CUDA_HOME:-}" ]; then
    echo "Run 'source ${HERE}/env.sh' first (VIRTUAL_ENV/CUDA_HOME unset)." >&2
    exit 2
fi

# Never contact Weights & Biases from a 5070 gate run.
export WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_API_KEY

mkdir -p "${OUT}"
echo "arm=${ARM} epochs=${EPOCHS} seed=${SEED}"
echo "out=${OUT}"

python3 "${HERE}/sysinfo.py" -o "${OUT}/sysinfo.json" >/dev/null
nvidia-smi > "${OUT}/nvidia_smi_before.txt" 2>&1

TIME_BIN=""
[ -x /usr/bin/time ] && TIME_BIN="/usr/bin/time -v -o ${OUT}/host_resources.txt"

# train_inr_unsup_spiral.py writes to './log/<tag>_<timestamp>' relative to the
# CURRENT WORKING DIRECTORY (not to the script directory), so running it from
# ${OUT} is what pins the 5070 outputs into their own tree.  Data paths are
# resolved against the script directory, so this is safe.
(
    cd "${OUT}" || exit 3
    # shellcheck disable=SC2086
    ${TIME_BIN} python3 -u "${ROOT}/train_inr_unsup_spiral.py" \
        --gpu 0 \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --data_path "${ROOT}/gassp1_data.mat" \
        --summary_epoch 16 \
        --kb_grid_size "${KB_GRID:-324}" \
        --tag "smoke_${ARM}" \
        "${ARM_ARGS[@]}"
) 2>&1 | tee "${OUT}/train_stdout.log"
train_status="${PIPESTATUS[0]}"

nvidia-smi > "${OUT}/nvidia_smi_after.txt" 2>&1

if [ "${train_status}" -ne 0 ]; then
    echo "TRAINING EXITED ${train_status} -- see ${OUT}/train_stdout.log" >&2
    exit "${train_status}"
fi

LOGDIR="$(find "${OUT}/log" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)"
echo "run directory: ${LOGDIR}"
python3 "${HERE}/check_run.py" "${LOGDIR}" --expect_epochs "${EPOCHS}" \
        | tee "${OUT}/acceptance.txt"
exit "${PIPESTATUS[0]}"
