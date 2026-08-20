#!/usr/bin/env bash
# run.sh -- 按 config 跑一个实验；断了用同一条命令续跑。
#
#   source tools/env.sh
#   bash tools/run.sh configs/u0.yaml            # 开跑，或从上次断点继续
#   bash tools/run.sh configs/u0.yaml --fresh    # 忽略断点，重头开始
#
# 产物在 $T1_RUNS/<name>/（默认 ~/runs），不进仓库。
# 跑完自动把文本摘要放进 reports/<name>/，那部分是要 commit 的 —— 见 tools/publish.sh。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "${HERE}/.." && pwd)"
CFG="${1:?用法: bash tools/run.sh configs/xxx.yaml [--fresh]}"; shift || true
FRESH=0; for a in "$@"; do [ "$a" = "--fresh" ] && FRESH=1; done
[ -f "${CFG}" ] || { echo "找不到 ${CFG}" >&2; exit 1; }
[ -n "${VIRTUAL_ENV:-}" ] || { echo "先 source tools/env.sh" >&2; exit 2; }
[ "${PYTORCH_CUDA_ALLOC_CONF:-}" = "expandable_segments:False" ] || {
  echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-<未设>}，不是实测有效的值。" >&2
  echo "重新 source tools/env.sh（必要时先 unset PYTORCH_CUDA_ALLOC_CONF）" >&2; exit 2; }

eval "$(python3 - "${CFG}" <<'PY'
import ast, shlex, sys
d = {}
for line in open(sys.argv[1]):
    line = line.split('#')[0].rstrip()
    if not line or ':' not in line: continue
    k, v = line.split(':', 1); k, v = k.strip(), v.strip()
    d[k] = ast.literal_eval(v) if v.startswith('[') else v
name = d.pop('name')
extra = d.pop('extra', [])
args = []
for k, v in d.items():
    args += ['--' + k, str(v)]
print(f"NAME={shlex.quote(name)}")
print(f"ARGS={shlex.quote(' '.join(args + [str(x) for x in extra]))}")
PY
)"

RUNS="${T1_RUNS:-${HOME}/runs}"; OUT="${RUNS}/${NAME}"; mkdir -p "${OUT}"
LOGDIR="$(find "${OUT}/log" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -1)"
RESUME=()
if [ "${FRESH}" -eq 0 ] && [ -n "${LOGDIR}" ] && [ -f "${LOGDIR}/ckpt.pt" ]; then
  RESUME=(--resume "${LOGDIR}")
  echo "发现断点，从 ${LOGDIR} 继续"
fi

echo "实验 ${NAME}   产物 ${OUT}"
echo "参数 ${ARGS} ${RESUME[*]:-}"
# shellcheck disable=SC2086
( cd "${OUT}" && python3 -u "${ROOT}/train_inr_unsup_spiral.py" \
    --gpu 0 --data_path "${T1_DATA:-${HOME}/data/gassp1_data.mat}" \
    --tag "${NAME}" ${ARGS} "${RESUME[@]}" ) 2>&1 | tee -a "${OUT}/train.log"
rc="${PIPESTATUS[0]}"
[ "${rc}" -ne 0 ] && { echo "训练退出码 ${rc}。断点还在，重跑同一条命令即可续上。" >&2; exit "${rc}"; }

LOGDIR="$(find "${OUT}/log" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)"
python3 "${HERE}/check_run.py" "${LOGDIR}" --expect_epochs "$(grep -oP '(?<=--epochs )\d+' <<< "${ARGS}")" \
  | tee "${OUT}/acceptance.txt"
echo
echo "下一步：bash tools/publish.sh ${NAME}    # 把文本结果提交进仓库"
