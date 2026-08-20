#!/usr/bin/env bash
# env.sh -- one place for the RTX 5070 / WSL2 session environment.
#
# Usage (must be SOURCED, not executed):
#     source tools/env.sh
#
# Optional: make it automatic for every new WSL shell by appending this line
# to ~/.bashrc  (edit the path to match your checkout):
#     source "$HOME/T1_blood_INR_5070_migration_20260817/tools/env.sh"
#
# Replaces the ad-hoc block in SETTING_PROGRESS_AND_TEST_PLAN_20260819.md sec.2.
# Idempotent: re-sourcing does not duplicate PATH entries.

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "env.sh must be sourced:  source ${0}" >&2
    exit 1
fi

T1_VENV="${T1_VENV:-$HOME/.venvs/t1_5070}"
T1_CUDA_HOME="${T1_CUDA_HOME:-/usr/local/cuda-12.8}"

_t1_prepend() {  # _t1_prepend VARNAME DIR  -- prepend only if not already present
    local var="$1" dir="$2" cur="${!1:-}"
    case ":${cur}:" in
        *":${dir}:"*) ;;
        *) export "${var}=${dir}${cur:+:${cur}}" ;;
    esac
}

if [ ! -x "${T1_VENV}/bin/python3" ]; then
    echo "env.sh: venv not found at ${T1_VENV}" >&2
    echo "             set T1_VENV=/path/to/venv before sourcing" >&2
    return 1
fi
if [ ! -d "${T1_CUDA_HOME}" ]; then
    echo "env.sh: CUDA toolkit not found at ${T1_CUDA_HOME}" >&2
    echo "             set T1_CUDA_HOME=/usr/local/cuda-XX.Y before sourcing" >&2
    return 1
fi

# shellcheck disable=SC1091
[ -z "${VIRTUAL_ENV:-}" ] && source "${T1_VENV}/bin/activate"

export CUDA_HOME="${T1_CUDA_HOME}"
_t1_prepend PATH            "${CUDA_HOME}/bin"
_t1_prepend LD_LIBRARY_PATH "${CUDA_HOME}/lib64"
_t1_prepend LD_LIBRARY_PATH "/usr/lib/wsl/lib"     # WSL libcuda.so.1 lives here
_t1_prepend LIBRARY_PATH    "${CUDA_HOME}/lib64"
_t1_prepend LIBRARY_PATH    "/usr/lib/wsl/lib"

# PyTorch 的 expandable_segments 用的是 CUDA 虚拟内存 API（cuMemCreate/cuMemMap/
# cuMemSetAccess），tiny-cuda-nn 的显存池用的是同一套。在 WSL2 + sm_120 + 610.x 驱动上
# 两者相争，会在第 2 步的 backward 抛 cuMemSetAccess: CUDA_ERROR_NOT_READY。
#
# 实测（tools/autopilot.sh，2026-08-20，真实数据真实规模）：
#   expandable_segments:True   0/4  全部崩在第 2 步
#   expandable_segments:False  12/12 通过；32 步 Gate B 一次过；200 步耐久一次过
#
# train_inr_unsup_spiral.py 里 setdefault 的 True 是 T4/Linux 时代的设置，
# 在 WSL 上是反的。这里显式覆盖它。
#
# 注意：这个结论是针对当前配置（kb_grid_size 324、log2_hashmap_size 24）验证的。
# 早先用 -hs 19 或 kb_grid 216 时同样关掉 VMM 仍会崩 —— 故障与具体分配尺寸有关。
# 改动模型规模或算子网格后，需要重跑 tools/autopilot.sh 重新确认。
# 这里必须【强制赋值】，不能写成 ${PYTORCH_CUDA_ALLOC_CONF:-...}：
# 只要当前 shell 里已经有旧值（比如之前 source 过老版本的本文件），:- 就会保留旧值，
# 修复完全不生效，而且从输出上看不出来。要临时覆盖请用 T1_ALLOC_CONF。
export PYTORCH_CUDA_ALLOC_CONF="${T1_ALLOC_CONF:-expandable_segments:False}"

# Nothing in the 5070 gate sequence is allowed to talk to Weights & Biases.
export WANDB_MODE="disabled"
export WANDB_DISABLED="true"

unset -f _t1_prepend
echo "env: venv=${VIRTUAL_ENV}"
echo "     CUDA_HOME=${CUDA_HOME}"
echo "     PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}   <- 必须是 expandable_segments:False"
echo "     wandb=${WANDB_MODE}"
if [ "${PYTORCH_CUDA_ALLOC_CONF}" != "expandable_segments:False" ]; then
    echo "     !! 警告：不是实测有效的那个值，训练会在第 2 步崩" >&2
fi
