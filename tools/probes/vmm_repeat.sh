#!/usr/bin/env bash
# probes/vmm_repeat.sh -- 同一配置重复多次，判断 cuMemSetAccess 崩溃是随机的还是确定的。
#
#   source tools/env.sh
#   bash   tools/probes/vmm_repeat.sh            # 两个配置各 5 次
#   TRIALS=8 bash tools/probes/vmm_repeat.sh
#
# 为什么需要这个：上一版 probes/vmm_screen.sh 每个配置只跑了 1 次。NVIDIA 论坛
# 说这个错误是 intermittent（间歇性）的，对间歇性故障来说 n=1 基本没有信息量 ——
# 「配置 2 通过」既可能是真的有效，也可能只是那一次运气好。必须用重复来区分：
#
#   A 稳定失败 + B 稳定通过   -> expandable_segments:False 是真解，固化即可
#   两个都是时好时坏          -> 纯随机，需要重试机制或换驱动，跟参数无关
#   通过率介于中间            -> 概率性，可以量化「一次 1600 步能跑完的机率」
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${T1_OUT_ROOT:-${HOME}/runs}/vmm_repeat_${STAMP}"
TRIALS="${TRIALS:-5}"
EPOCHS="${EPOCHS:-4}"
mkdir -p "${OUT}"
REPORT="${OUT}/REPORT.txt"
[ -n "${VIRTUAL_ENV:-}" ] || { echo "先 source env.sh" >&2; exit 2; }
export WANDB_MODE=disabled WANDB_DISABLED=true

{ echo "cuMemSetAccess 崩溃：随机性检验  ${STAMP}"
  echo "每个配置 ${TRIALS} 次，每次 ${EPOCHS} 步"
  echo "起始显存: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
  echo; } | tee "${REPORT}"

trial_set () {                          # trial_set <标签> <ALLOC_CONF> <额外参数...>
  local label="$1" conf="$2"; shift 2
  local pass=0 fail=0 detail=""
  printf '── %s   (PYTORCH_CUDA_ALLOC_CONF=%s %s)\n' "${label}" "${conf}" "$*" | tee -a "${REPORT}"
  for i in $(seq 1 "${TRIALS}"); do
    local dir="${OUT}/${label}_t${i}"; mkdir -p "${dir}"
    ( cd "${dir}" && PYTORCH_CUDA_ALLOC_CONF="${conf}" python3 -u \
        "${ROOT}/train_inr_unsup_spiral.py" --gpu 0 --epochs "${EPOCHS}" --seed 0 \
        --data_path "${ROOT}/gassp1_data.mat" --summary_epoch 10000 \
        --tag "rep_${label}_t${i}" "$@" ) > "${dir}/stdout.log" 2>&1
    if [ $? -eq 0 ]; then
      pass=$((pass+1)); detail+="  第${i}次 通过"
    else
      fail=$((fail+1))
      local at; at="$(grep -oE "[0-9]+/${EPOCHS} \[" "${dir}/stdout.log" | tail -1 | cut -d/ -f1)"
      detail+="  第${i}次 失败(第${at:-0}步)"
    fi
    printf '.' 
  done
  printf '\n'
  printf '   通过 %d/%d\n  %s\n\n' "${pass}" "${TRIALS}" "${detail}" | tee -a "${REPORT}"
}

trial_set A "expandable_segments:True"  --kb_grid_size 324
trial_set B "expandable_segments:False" --kb_grid_size 324

# 关键观察：5 次失败全部发生在第 2 步的 backward，从来不是第 3、4 步。
# 这不像均匀随机，更像是显存池的「第二次扩容」这一个特定事件在出问题。
# 如果确实如此，那么只要熬过前几步，后面就是稳的 —— 那么「进程级重试」就是
# 一个可用的权宜之计。下面直接验证这一点。
echo "── 阶段 2：熬过前几步之后还稳不稳？（32 步，最多重试 5 次）" | tee -a "${REPORT}"
survived=0
for i in $(seq 1 5); do
  dir="${OUT}/long32_t${i}"; mkdir -p "${dir}"
  ( cd "${dir}" && PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False" python3 -u \
      "${ROOT}/train_inr_unsup_spiral.py" --gpu 0 --epochs 32 --seed 0 \
      --data_path "${ROOT}/gassp1_data.mat" --kb_grid_size 324 \
      --summary_epoch 16 --tag "rep_long32_t${i}" ) > "${dir}/stdout.log" 2>&1
  rc=$?
  at="$(grep -oE '[0-9]+/32 \[' "${dir}/stdout.log" | tail -1 | cut -d/ -f1)"
  pk="$(grep -oE 'peak=[0-9.]+' "${dir}/stdout.log" | tail -1)"
  if [ "${rc}" -eq 0 ]; then
    printf '   第%d次尝试：完成全部 32 步  %s\n' "${i}" "${pk}" | tee -a "${REPORT}"
    printf '   >>> 32 步冒烟测试成功，日志目录: %s/log\n\n' "${dir}" | tee -a "${REPORT}"
    survived=1; break
  else
    printf '   第%d次尝试：失败于第 %s 步  %s\n' "${i}" "${at:-0}" "${pk}" | tee -a "${REPORT}"
  fi
done
[ "${survived}" -eq 0 ] && printf '   5 次都没跑完 32 步。\n\n' | tee -a "${REPORT}"

{ echo "=============================================="
  echo "怎么读："
  echo "  A 全失败 且 B 全通过     -> expandable_segments:False 是真解。固化进 env.sh。"
  echo "  两边都时好时坏           -> 纯随机的驱动 bug，跟参数无关。走重试或换驱动。"
  echo "  B 部分通过               -> 概率性。但如果失败永远只发生在最初几步、"
  echo "                              一旦熬过就稳定，那么进程级重试就够用（阶段 2 验证的就是这个）。"
  echo "  阶段 2 成功              -> 32 步冒烟测试实际上已经完成，可以拿它当 Gate B 的产物。"
  echo
  echo "完整输出: ${OUT}"; } | tee -a "${REPORT}"

if [ -f "${HERE}/.upstream" ]; then
  RS="$(head -1 "${HERE}/.upstream")"
  [ -d "${RS}" ] && { mkdir -p "${RS}/_gate_reports/vmm_repeat_${STAMP}"
    cp "${REPORT}" "${RS}/_gate_reports/vmm_repeat_${STAMP}/" 2>/dev/null
    echo "报告已回写到 ${RS}/_gate_reports/vmm_repeat_${STAMP}/"; }
fi
echo "报告: ${REPORT}"
