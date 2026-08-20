#!/usr/bin/env bash
# doctor.sh -- Gate A: environment re-verification in a fresh WSL session.
#
#     source tools/env.sh
#     bash   tools/doctor.sh
#
# Verifies ONLY the environment. It does not train and produces no scientific
# result. Writes gate_a_<timestamp>/ with the runtime signature and all output.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${T1_OUT_ROOT:-${ROOT}/runs}/doctor_${STAMP}"
mkdir -p "${OUT}"

log() { echo -e "\n=== $* ===" | tee -a "${OUT}/gate_a.log"; }
run() { echo "\$ $*" | tee -a "${OUT}/gate_a.log"; "$@" 2>&1 | tee -a "${OUT}/gate_a.log"; return "${PIPESTATUS[0]}"; }

status=0

log "0. context"
run bash -c 'echo "cwd=$(pwd)"; echo "python=$(command -v python3)"; echo "CUDA_HOME=${CUDA_HOME:-<unset>}"'
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "WARNING: no virtualenv active -- did you 'source env.sh' first?" | tee -a "${OUT}/gate_a.log"
fi

log "1. verify_migration.py  (local files vs manifest SHA-256)"
( cd "${ROOT}" && run python3 verify_migration.py ) || status=1

log "2. verify_migration.py --env  (GPU / torch / CUDA / torchkbnufft / tinycudann)"
( cd "${ROOT}" && run python3 verify_migration.py --env ) || status=1

log "3. deps.py  (the imports verify_migration does NOT cover)"
( cd "${ROOT}" && run python3 "${HERE}/deps.py" ) || status=1

log "4. probes/tcnn.py  (真正跑一次 tcnn 内核 -- import 成功不代表内核存在)"
( cd "${ROOT}" && run python3 "${HERE}/probes/tcnn.py" ) || status=1

log "5. runtime signature"
( cd "${ROOT}" && run python3 "${HERE}/sysinfo.py" -o "${OUT}/sysinfo.json" ) || status=1

log "6. host memory headroom (system RAM is the tighter budget: ~7.3 GiB in WSL)"
run bash -c 'free -h; echo; nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv'

log "RESULT"
if [ "${status}" -eq 0 ]; then
    echo "GATE A PASSED -- environment only. No training has been run." | tee -a "${OUT}/gate_a.log"
else
    echo "GATE A FAILED -- fix the items above before Gate B." | tee -a "${OUT}/gate_a.log"
fi
echo "artifacts: ${OUT}" | tee -a "${OUT}/gate_a.log"
exit "${status}"
