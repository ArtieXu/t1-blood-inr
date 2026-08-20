#!/usr/bin/env bash
# probes/vmm_screen.sh -- 定位 Gate B 的 cuMemSetAccess / CUDA_ERROR_NOT_READY 崩溃。
#
#   source tools/env.sh
#   bash   tools/probes/vmm_screen.sh
#
# 背景：tiny-cuda-nn 用 CUDA 虚拟内存管理 API（cuMemCreate/cuMemMap/cuMemSetAccess）
# 管自己的显存池。PyTorch 的 expandable_segments:True 用的是**同一套 API**。
# NVIDIA 论坛已确认 WSL2 + sm_120 + 610.x 驱动上 cuMemSetAccess 会间歇性返回
# CUDA_ERROR_NOT_READY，且是在显存池向上增长、逼近设备上限时发生。
#
# 本脚本跑 5 个 4 步的短配置，每个约半分钟，看哪一个能活下来：
#   1 控制组：现状（expandable_segments:True, kb_grid 324）—— 预期 FAIL
#   2 关掉 torch 的 VMM 分配器             <- 主假设
#   3 在 2 的基础上把 hash 表缩到 2^19    <- 若只有它活，说明是显存压力
#   4 在 2 的基础上把 kb_grid 降到 216    <- 减小 torch 侧张量
#   5 只降 kb_grid，不关 VMM               <- 拆开两个变量
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${T1_OUT_ROOT:-${HOME}/runs}/vmm_diag_${STAMP}"
EPOCHS="${EPOCHS:-4}"
mkdir -p "${OUT}"
REPORT="${OUT}/REPORT.txt"

[ -n "${VIRTUAL_ENV:-}" ] || { echo "先 source env.sh" >&2; exit 2; }
export WANDB_MODE=disabled WANDB_DISABLED=true

{
  echo "tcnn VMM 崩溃诊断  ${STAMP}"
  echo "每个配置 ${EPOCHS} 步。关注：能不能活过第 2 步的 backward。"
  echo
  echo "开跑前的显存占用（WSL 里看不到 Windows 侧进程，这部分通常是桌面/浏览器）："
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  echo
} | tee "${REPORT}"

run_case () {                       # run_case <序号> <说明> <ALLOC_CONF> <额外参数...>
  local n="$1" desc="$2" conf="$3"; shift 3
  local dir="${OUT}/case${n}"; mkdir -p "${dir}"
  printf '%s\n' "── 配置 ${n}: ${desc}" | tee -a "${REPORT}"
  printf '   PYTORCH_CUDA_ALLOC_CONF=%s  额外参数: %s\n' "${conf}" "$*" | tee -a "${REPORT}"
  (
    cd "${dir}" || exit 3
    PYTORCH_CUDA_ALLOC_CONF="${conf}" python3 -u "${ROOT}/train_inr_unsup_spiral.py" \
      --gpu 0 --epochs "${EPOCHS}" --seed 0 \
      --data_path "${ROOT}/gassp1_data.mat" \
      --summary_epoch 10000 --tag "vmm_case${n}" "$@"
  ) > "${dir}/stdout.log" 2>&1
  local rc=$?
  local steps peak err
  steps="$(grep -oE '[0-9]+/'"${EPOCHS}"' \[' "${dir}/stdout.log" | tail -1 | cut -d/ -f1)"
  peak="$(grep -oE 'peak=[0-9.]+' "${dir}/stdout.log" | tail -1)"
  if [ "${rc}" -eq 0 ]; then
    printf '   => 通过   完成 %s/%s 步   %s\n\n' "${EPOCHS}" "${EPOCHS}" "${peak:-peak=?}" | tee -a "${REPORT}"
  else
    err="$(grep -oE '(RuntimeError|torch\.OutOfMemoryError|CUDA error).*' "${dir}/stdout.log" | tail -1 | cut -c1-150)"
    printf '   => 失败   跑到第 %s 步   %s\n      %s\n\n' "${steps:-0}" "${peak:-peak=?}" "${err:-未捕获到错误行}" | tee -a "${REPORT}"
  fi
  return 0
}

run_case 1 "控制组：现状"                  "expandable_segments:True"  --kb_grid_size 324
run_case 2 "关掉 torch 的 VMM 分配器"       "expandable_segments:False" --kb_grid_size 324
run_case 3 "关 VMM + hash 表 2^19"          "expandable_segments:False" --kb_grid_size 324 -hs 19
run_case 4 "关 VMM + kb_grid 216"           "expandable_segments:False" --kb_grid_size 216
run_case 5 "只降 kb_grid 216，不关 VMM"     "expandable_segments:True"  --kb_grid_size 216

{
  echo "=============================================="
  echo "怎么读这份结果："
  echo "  只有 1 失败、2 通过        -> torch 与 tcnn 抢 VMM。把 expandable_segments:False"
  echo "                               固化进 env.sh 即可，不必改任何科学参数。"
  echo "  1、2 都失败、3 通过        -> 是显存压力。先关掉占显存的 Windows 程序再试；"
  echo "                               仍不行则需要分块渲染（改代码）。"
  echo "  4/5 的差异                 -> 说明 torch 侧张量大小也参与其中。"
  echo "  全部失败                   -> 驱动层面的问题。回退到文档记录的 576.88 驱动，"
  echo "                               或在 tcnn 的 cuMemSetAccess 处加重试补丁。"
  echo
  echo "完整输出: ${OUT}/case*/stdout.log"
} | tee -a "${REPORT}"

# 把报告回写到源目录，方便直接读
if [ -f "${HERE}/.upstream" ]; then
  RS="$(head -1 "${HERE}/.upstream")"
  if [ -d "${RS}" ]; then
    mkdir -p "${RS}/_gate_reports/vmm_diag_${STAMP}"
    cp "${REPORT}" "${RS}/_gate_reports/vmm_diag_${STAMP}/" 2>/dev/null
    for d in "${OUT}"/case*; do
      mkdir -p "${RS}/_gate_reports/vmm_diag_${STAMP}/$(basename "$d")"
      tail -40 "$d/stdout.log" > "${RS}/_gate_reports/vmm_diag_${STAMP}/$(basename "$d")/stdout_tail.log" 2>/dev/null
    done
    echo "报告已回写到 ${RS}/_gate_reports/vmm_diag_${STAMP}/"
  fi
fi
echo "报告: ${REPORT}"
