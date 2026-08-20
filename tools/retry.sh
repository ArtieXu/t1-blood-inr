#!/usr/bin/env bash
# retry.sh -- 给 train_inr_unsup_spiral.py 套一层重试外壳。
#
#   source tools/env.sh
#   bash tools/retry.sh -- --epochs 32 --seed 0 --kb_grid_size 324
#
# 只在遇到 WSL 的 cuMemSetAccess / CUDA_ERROR_NOT_READY 这类瞬态驱动故障时重试。
# 其它错误（OOM、参数错误、数据问题）直接退出 —— 那些重试也没用，重试只会掩盖问题。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
MAX="${MAX_RETRY:-10}"
[ "${1:-}" = "--" ] && shift
[ -n "${VIRTUAL_ENV:-}" ] || { echo "先 source env.sh" >&2; exit 2; }

for t in $(seq 1 "${MAX}"); do
  echo "── 第 ${t}/${MAX} 次尝试"
  python3 -u "${ROOT}/train_inr_unsup_spiral.py" "$@" 2>&1 | tee /tmp/rwr_$$.log
  rc="${PIPESTATUS[0]}"
  [ "${rc}" -eq 0 ] && { echo "成功（第 ${t} 次）"; rm -f /tmp/rwr_$$.log; exit 0; }
  if grep -q "CUDA_ERROR_NOT_READY\|cuMemSetAccess\|cuMemMap" /tmp/rwr_$$.log; then
    echo "命中 WSL 的瞬态 VMM 故障，重试。"; sleep 3; continue
  fi
  echo "不是瞬态故障（退出码 ${rc}），不重试。"; rm -f /tmp/rwr_$$.log; exit "${rc}"
done
echo "重试 ${MAX} 次仍未成功。"; rm -f /tmp/rwr_$$.log; exit 1
