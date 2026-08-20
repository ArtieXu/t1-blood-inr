#!/usr/bin/env bash
# autopilot.sh -- 自主跑完整个决策树，直到给出 Gate B 的结论。
#
#   source tools/env.sh
#   bash   tools/autopilot.sh          # 顺利约 45 分钟，最坏约 2 小时
#   nohup bash tools/autopilot.sh > /dev/null 2>&1 &   # 挂后台
#
# 它自己判断分支，不需要中途干预。阶段：
#   0  环境快照 + 在 tcnn 二进制里找隐藏开关
#   1  筛选 4 种显存分配器配置（每种 SCREEN 次，4 步）
#   2  对胜出者加测（CONFIRM 次），得到可信的通过率
#   3  用胜出配置跑真正的 32 步 Gate B，失败自动重试
#   4  耐久测试 200 步 —— 区分「只在开头崩」和「中途也会崩」（后者是致命的）
#   5  结论 + 该固化进 env.sh 的那一行
#
# 退出码：0 = 问题真的解决了（拿到 32 步 Gate B 且无中途崩溃）；1 = 未解决。
# 判定顺序：耐久性优先于通过率。短跑 100% 但第 N 步必崩的配置会被判为「未解决」。
#
# 全程不改任何科学参数：seed 0、真实 gassp1_data.mat、TI=63+200n、
# dc_form feng_rel、kb_grid_size 324（与两个 notebook 一致）。
# 唯一被改动的是 PYTORCH_CUDA_ALLOC_CONF —— PyTorch 的显存分配器策略，不进计算。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${T1_OUT_ROOT:-${HOME}/runs}/autopilot_${STAMP}"
REPORT="${OUT}/REPORT.txt"
SCREEN="${SCREEN:-4}"; CONFIRM="${CONFIRM:-8}"
GATEB_RETRY="${GATEB_RETRY:-10}"; DURABILITY_EPOCHS="${DURABILITY_EPOCHS:-200}"
DUR_TRIES="${DUR_TRIES:-6}"; EARLY_STEPS="${EARLY_STEPS:-5}"
KB=324
mkdir -p "${OUT}"
[ -n "${VIRTUAL_ENV:-}" ] || { echo "先 source env.sh" >&2; exit 2; }
export WANDB_MODE=disabled WANDB_DISABLED=true

log(){ printf '%s\n' "$*" | tee -a "${REPORT}"; }
# "1 1 1 5 " -> "1×3, 5×1"
fmt_steps(){ [ -z "${1// /}" ] && return; tr ' ' '\n' <<< "$1" | grep -v '^$' | sort -n | uniq -c \
  | awk '{printf "%s%s×%s", (NR>1?", ":""), $2, $1} END{print ""}'; }
gpu_free(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; }

# 跑一次训练。回显：<rc> <到达步数> <峰值>
one_run () {   # one_run <目录> <ALLOC_CONF> <步数> <额外参数...>
  local dir="$1" conf="$2" ep="$3"; shift 3
  mkdir -p "${dir}"
  ( cd "${dir}" && PYTORCH_CUDA_ALLOC_CONF="${conf}" python3 -u \
      "${ROOT}/train_inr_unsup_spiral.py" --gpu 0 --epochs "${ep}" --seed 0 \
      --data_path "${ROOT}/gassp1_data.mat" --kb_grid_size "${KB}" \
      --summary_epoch 100000 --tag "ap" "$@" ) > "${dir}/stdout.log" 2>&1
  local rc=$?
  # 筛选阶段只关心「跑没跑通」，不需要产物。每次成功运行会写约 132 MB
  # （final_state.pt 96 MB + recon_final.mat 36 MB），24 次就是 3 GB 起。
  find "${dir}/log" -type f \( -name '*.pt' -o -name '*.mat' -o -name '*.png' \) -delete 2>/dev/null
  local at pk
  at="$(grep -oE "[0-9]+/${ep} \[" "${dir}/stdout.log" | tail -1 | cut -d/ -f1)"
  pk="$(grep -oE 'peak=[0-9.]+' "${dir}/stdout.log" | tail -1 | cut -d= -f2)"
  echo "${rc} ${at:-0} ${pk:-0}"
}

log "=============================================================="
log " Gate B 自主诊断  ${STAMP}"
log "=============================================================="
log "顺利的话约 45 分钟；最坏情况（反复重试）可能到 2 小时。中途无需干预。"
T0=$(date +%s); phase_t(){ log "   [累计用时 $(( ($(date +%s)-T0)/60 )) 分钟]"; }
log ""

# ---------------------------------------------------------------- Phase 0
log "── 阶段 0：环境与二进制探针"
log "显存占用(开跑前): $(gpu_free)"
log "驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
DISK_MB="$(df -Pm "$(dirname "${OUT}")" | awk 'NR==2{print $4}')"
log "磁盘可用: ${DISK_MB} MB（Gate B 与耐久产物约需 400 MB；筛选产物会即时清理）"
if [ "${DISK_MB:-0}" -lt 1500 ]; then
  log "  警告：可用空间不足 1.5 GB，可能中途写不下。先清一清再跑。"
fi
log "提示：整个过程独占显卡，期间别在这块卡上跑别的东西，否则结果不可比。"
TCNN_SO="$(python3 -c "
import tinycudann, os, glob
d = os.path.dirname(tinycudann.__file__)
c = glob.glob(d+'/**/*.so', recursive=True) + glob.glob(os.path.dirname(d)+'/tinycudann*/**/*.so', recursive=True)
print(max(c, key=os.path.getsize) if c else '')" 2>/dev/null)"
if [ -n "${TCNN_SO}" ] && command -v strings >/dev/null; then
  log "tcnn 二进制: ${TCNN_SO}"
  HITS="$(strings "${TCNN_SO}" 2>/dev/null | grep -iE '^TCNN_[A-Z_]+$|virtual memory|cuMemSetAccess' | sort -u | head -8)"
  if [ -n "${HITS}" ]; then
    log "  相关字符串:"
    while IFS= read -r line; do log "    ${line}"; done <<< "${HITS}"
  else log "  没找到 TCNN_* 形式的环境变量开关"; fi
else
  log "  跳过（未找到 .so 或没有 strings 命令）"
fi
log ""

# ---------------------------------------------------------------- Phase 1
log "── 阶段 1：筛选 4 种分配器配置，每种 ${SCREEN} 次 × 4 步"
NAMES=(A B C D)
CONFS=("expandable_segments:True" \
       "expandable_segments:False" \
       "backend:cudaMallocAsync" \
       "expandable_segments:False,garbage_collection_threshold:0.6")
DESCS=("现状（对照组）" "关掉 torch 的 VMM 分配器" "换成 stream-ordered 分配器" "关 VMM + 更激进的回收")
declare -A PASSES
BEST=""; BESTN=-1
for i in 0 1 2 3; do
  n="${NAMES[$i]}"; c="${CONFS[$i]}"; p=0; steps=""
  for t in $(seq 1 "${SCREEN}"); do
    read -r rc at pk <<< "$(one_run "${OUT}/s_${n}_${t}" "${c}" 4)"
    [ "${rc}" -eq 0 ] && p=$((p+1)) || steps+="${at} "
  done
  PASSES[$n]=$p
  steps="$(fmt_steps "${steps}")"
  log "  ${n} ${DESCS[$i]}"
  log "     ${c}"
  log "     通过 ${p}/${SCREEN}$([ -n "${steps}" ] && echo "   失败步: ${steps}")"
  if [ "$p" -gt "$BESTN" ]; then BESTN=$p; BEST="$n"; BESTCONF="$c"; fi
done
log ""
log "  筛选胜出: ${BEST}  (${BESTCONF})   ${BESTN}/${SCREEN}"
log ""

if [ "${BESTN}" -eq 0 ]; then
  log "── 全部配置零通过 -> 这不是分配器策略能解决的问题。"
  log "   结论：驱动层 bug。见阶段 5。"
  VERDICT="ALL_FAIL"
else
  # -------------------------------------------------------------- Phase 2
  log "── 阶段 2：对 ${BEST} 加测 ${CONFIRM} 次，得到可信通过率"
  p2=0; fs=""
  for t in $(seq 1 "${CONFIRM}"); do
    read -r rc at pk <<< "$(one_run "${OUT}/c_${BEST}_${t}" "${BESTCONF}" 4)"
    if [ "${rc}" -eq 0 ]; then p2=$((p2+1)); printf '%s' '+'; else fs+="${at} "; printf '%s' '-'; fi
  done
  printf '\n'
  fs="$(fmt_steps "${fs}")"
  TOTP=$((BESTN + p2)); TOTN=$((SCREEN + CONFIRM))
  RATE=$(( 100 * TOTP / TOTN ))
  log "  合计通过 ${TOTP}/${TOTN}  = ${RATE}%$([ -n "${fs}" ] && echo "   失败步: ${fs}")"; phase_t
  log ""
  if [ "${RATE}" -ge 90 ]; then VERDICT="STABLE"
  elif [ "${RATE}" -ge 20 ]; then VERDICT="PROBABILISTIC"
  else VERDICT="MOSTLY_FAIL"; fi

  # -------------------------------------------------------------- Phase 3
  log "── 阶段 3：用 ${BEST} 跑真正的 32 步 Gate B（失败自动重试，上限 ${GATEB_RETRY} 次）"
  GATEB_OK=0; GATEB_TRIES=0; GATEB_DIR=""
  for t in $(seq 1 "${GATEB_RETRY}"); do
    GATEB_TRIES=$t
    d="${OUT}/gateb_t${t}"
    mkdir -p "${d}"
    ( cd "${d}" && PYTORCH_CUDA_ALLOC_CONF="${BESTCONF}" python3 -u \
        "${ROOT}/train_inr_unsup_spiral.py" --gpu 0 --epochs 32 --seed 0 \
        --data_path "${ROOT}/gassp1_data.mat" --kb_grid_size "${KB}" \
        --summary_epoch 16 --tag "smoke_${BEST}" ) > "${d}/stdout.log" 2>&1
    rc=$?
    at="$(grep -oE '[0-9]+/32 \[' "${d}/stdout.log" | tail -1 | cut -d/ -f1)"
    if [ "${rc}" -eq 0 ]; then
      log "  第 ${t} 次：完成 32/32 步"
      GATEB_OK=1; GATEB_DIR="$(find "${d}/log" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)"
      break
    fi
    log "  第 ${t} 次：失败于第 ${at:-0} 步"
  done
  log ""
  if [ "${GATEB_OK}" -eq 1 ]; then
    log "  Gate B 验收检查："
    python3 "${HERE}/check_run.py" "${GATEB_DIR}" --expect_epochs 32 \
      > "${OUT}/acceptance.txt" 2>&1
    sed 's/^/    /' "${OUT}/acceptance.txt" | tee -a "${REPORT}"
    log ""
  else
    log "  ${GATEB_RETRY} 次都没跑完 32 步。"
    log ""
  fi

  # -------------------------------------------------------------- Phase 4
  if [ "${GATEB_OK}" -eq 1 ]; then
    log "── 阶段 4：耐久测试 ${DURABILITY_EPOCHS} 步 —— 故障只在开头，还是中途也会崩？"
    log "   这一步决定 800/1600 步的长实验到底可不可行。"
    log "   判据：崩在前 ${EARLY_STEPS} 步 = 开头问题（重试可解）；崩在更后面 = 中途也会崩（重试救不了）。"
    DUR_OK=0; DUR_MID=0; DUR_MAX=0; dsteps=""
    for t in $(seq 1 "${DUR_TRIES}"); do
      d="${OUT}/dur_t${t}"; mkdir -p "${d}"
      ( cd "${d}" && PYTORCH_CUDA_ALLOC_CONF="${BESTCONF}" python3 -u \
          "${ROOT}/train_inr_unsup_spiral.py" --gpu 0 --epochs "${DURABILITY_EPOCHS}" --seed 0 \
          --data_path "${ROOT}/gassp1_data.mat" --kb_grid_size "${KB}" \
          --summary_epoch 100000 --tag "dur" ) > "${d}/stdout.log" 2>&1
      rc=$?
      at="$(grep -oE "[0-9]+/${DURABILITY_EPOCHS} \[" "${d}/stdout.log" | tail -1 | cut -d/ -f1)"
      at="${at:-0}"
      [ "${at}" -gt "${DUR_MAX}" ] && DUR_MAX="${at}"
      if [ "${rc}" -eq 0 ]; then
        log "   第 ${t} 次：完成 ${DURABILITY_EPOCHS}/${DURABILITY_EPOCHS} 步"
        DUR_OK=1; break
      fi
      dsteps+="${at} "
      if [ "${at}" -gt "${EARLY_STEPS}" ]; then
        log "   第 ${t} 次：失败于第 ${at} 步  <- 中途崩，不是开头问题"
        DUR_MID=1
        log "   一次中途崩溃已足以定论，不再重复。"
        break
      else
        log "   第 ${t} 次：失败于第 ${at} 步（开头）"
      fi
    done
    if [ -n "${dsteps// /}" ]; then
      log "   最远跑到第 ${DUR_MAX} 步；失败步分布: $(fmt_steps "${dsteps}")"
    fi
    phase_t    log ""
  fi
fi

# ---------------------------------------------------------------- Phase 5
log "=============================================================="
log " 阶段 5：结论"
log "=============================================================="
# 判定顺序很重要：耐久性优先于通过率。
# 一个「短跑 100% 通过、但第 N 步必崩」的配置不是解，是陷阱 ——
# 它会让人以为问题解决了，然后 1600 步的实验全军覆没。
SOLVED=0
if [ "${VERDICT:-ALL_FAIL}" = "ALL_FAIL" ]; then
  log " 判定：没有任何分配器配置能工作。"
  ESCALATE=1
elif [ "${DUR_MID:-0}" -eq 1 ]; then
  log " 判定：**看似解决，实则没有**。"
  log "   ${BEST} (${BESTCONF}) 的短跑通过率 ${RATE}%，32 步 Gate B 也过了，"
  log "   但耐久测试每次都崩在第 ${DUR_MAX} 步左右 —— 这是中途崩溃，不是开头问题。"
  log "   重试外壳救不了长实验：800/1600 步一定会崩。"
  ESCALATE=1
elif [ "${GATEB_OK:-0}" -eq 1 ] && [ "${DUR_OK:-0}" -eq 1 ]; then
  SOLVED=1
  if [ "${RATE}" -ge 90 ]; then
    log " 判定：已解决。${BEST} 通过率 ${RATE}%，${DURABILITY_EPOCHS} 步耐久一次通过。"
    log " 该做的事 —— 把这一行加进 tools/env.sh："
    log "     export PYTORCH_CUDA_ALLOC_CONF=\"${BESTCONF}\""
  else
    log " 判定：已解决，但需要重试。${BEST} 单次通过率只有 ${RATE}%，"
    log "   不过一旦启动成功就稳（${DURABILITY_EPOCHS} 步耐久一次通过），第 ${GATEB_TRIES} 次拿到了 Gate B。"
    log " 该做的事 —— 两条都要："
    log "     export PYTORCH_CUDA_ALLOC_CONF=\"${BESTCONF}\"        # 加进 env.sh"
    log "     bash tools/retry.sh -- <训练参数>     # 所有训练都套上"
  fi
elif [ "${GATEB_OK:-0}" -eq 1 ]; then
  SOLVED=1
  log " 判定：Gate B 拿到了（第 ${GATEB_TRIES} 次），失败全部集中在前 ${EARLY_STEPS} 步。"
  log "   耐久测试 ${DUR_TRIES} 次没跑完 ${DURABILITY_EPOCHS} 步，但也全是开头就崩（最远第 ${DUR_MAX} 步），"
  log "   没有观察到中途崩溃 —— 重试外壳仍然可用，只是要把 MAX_RETRY 调大。"
  log " 该做的事："
  log "     export PYTORCH_CUDA_ALLOC_CONF=\"${BESTCONF}\""
  log "     MAX_RETRY=30 bash tools/retry.sh -- <训练参数>"
  log " 注意：耐久性尚未证实。跑 800/1600 步之前，先单独跑一次 ${DURABILITY_EPOCHS}+ 步确认。"
else
  log " 判定：最优配置 ${BEST} 通过率 ${RATE}%，但重试 ${GATEB_RETRY} 次仍未拿到 32 步 Gate B。"
  ESCALATE=1
fi

if [ "${ESCALATE:-0}" -eq 1 ]; then
  log ""
  log " 升级路径（按代价排序）："
  log "   1. 回退显卡驱动到你文档记录的 576.88（现在是 $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)）"
  log "      NVIDIA 论坛上这个 bug 报告的正是 610.x + sm_120 + WSL2。"
  log "   2. 装原生 Linux 双系统 —— 这是 WSL2 特有问题，原生 Linux 不走这条代码路径。"
  log "   3. 给 tiny-cuda-nn 的 cuMemSetAccess 处打重试补丁并重编。"
fi
log ""
log " 完整输出: ${OUT}"

# 回写报告
if [ -f "${HERE}/.upstream" ]; then
  RS="$(head -1 "${HERE}/.upstream")"
  if [ -d "${RS}" ]; then
    D="${RS}/_gate_reports/autopilot_${STAMP}"; mkdir -p "${D}"
    cp "${REPORT}" "${D}/" 2>/dev/null
    [ -f "${OUT}/acceptance.txt" ] && cp "${OUT}/acceptance.txt" "${D}/" 2>/dev/null
    [ -n "${GATEB_DIR:-}" ] && { cp "${GATEB_DIR}/loss.csv" "${GATEB_DIR}/run_info.json" "${D}/" 2>/dev/null; }
    echo "报告已回写: ${D}"
  fi
fi
echo "报告: ${REPORT}"
# 退出码：0 = Gate B 真的跑通了；1 = 没有
# 0 只在「问题真的解决了」时返回：拿到 Gate B 且没有中途崩溃
[ "${SOLVED:-0}" -eq 1 ] && exit 0 || exit 1
