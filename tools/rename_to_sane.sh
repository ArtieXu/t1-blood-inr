#!/usr/bin/env bash
# rename_to_sane.sh -- 一次性把工具目录改成正常的命名，并修好所有相互引用。
#
#   bash tools/rename_to_sane.sh --dry      # 预演，只打印会做什么
#   bash tools/rename_to_sane.sh            # 真的改
#
# 在项目根目录运行。如果已经是 git 仓库就用 git mv 保留历史。
# 改完会自检：所有 .sh 过 bash -n、所有 .py 过 ast.parse、全树 grep 残留旧名。
set -uo pipefail
ROOT="$(pwd)"; DRY=0; [ "${1:-}" = "--dry" ] && DRY=1
KIT_OLD="tools"; KIT_NEW="tools"
[ -d "${KIT_OLD}" ] || { echo "在项目根目录运行（找不到 ${KIT_OLD}/）" >&2; exit 1; }
GITMV=0; [ -d .git ] && GITMV=1

# 旧名 -> 新名（相对 tools/）。文档挪去 docs/。
MAP="
env.sh|env.sh
doctor.sh|doctor.sh
smoke.sh|smoke.sh
autopilot.sh|autopilot.sh
retry.sh|retry.sh
gates.sh|gates.sh
install_deps.sh|install_deps.sh
precommit.sh|precommit.sh
probes/vmm_screen.sh|probes/vmm_screen.sh
probes/vmm_repeat.sh|probes/vmm_repeat.sh
deps.py|deps.py
sysinfo.py|sysinfo.py
check_run.py|check_run.py
probes/tcnn.py|probes/tcnn.py
probes/nufft.py|probes/nufft.py
probes/subspace.py|probes/subspace.py
probes/subspace_preb0_ti50.csv|probes/subspace_preb0_ti50.csv
probes/subspace_preb0_ti50.json|probes/subspace_preb0_ti50.json
gitignore|gitignore
README.md|README.md
"
DOCS="
docs/RANK5_AUDIT_20260819.md|RANK5_AUDIT_20260819.md
docs/NEXT_STEPS_20260819.md|NEXT_STEPS_20260819.md
docs/BLOCKERS_20260819.md|BLOCKERS_20260819.md
docs/DRYRUN_20260820.md|DRYRUN_20260820.md
"
say(){ [ "${DRY}" -eq 1 ] && echo "  [预演] $*" || echo "  $*"; }
do_mv(){ # do_mv <源> <目标>
  [ -e "$1" ] || return 0
  say "mv $1 -> $2"
  [ "${DRY}" -eq 1 ] && return 0
  mkdir -p "$(dirname "$2")"
  if [ "${GITMV}" -eq 1 ] && git ls-files --error-unmatch "$1" >/dev/null 2>&1; then git mv -f "$1" "$2"
  else mv -f "$1" "$2"; fi
}

echo "── 1. 移动文件"
mkdir -p "${KIT_OLD}/probes" docs 2>/dev/null
while IFS='|' read -r o n; do [ -z "$o" ] && continue
  do_mv "${KIT_OLD}/${o}" "${KIT_OLD}/${n}"; done <<< "${MAP}"
while IFS='|' read -r o n; do [ -z "$o" ] && continue
  do_mv "${KIT_OLD}/${o}" "docs/${n}"; done <<< "${DOCS}"
do_mv "${KIT_OLD}/dryrun/run.py"    "${KIT_OLD}/dryrun/run.py"
do_mv "${KIT_OLD}/dryrun/fake_data.py" "${KIT_OLD}/dryrun/fake_data.py"
do_mv "${KIT_OLD}/dryrun/stub/tinycudann.py"   "${KIT_OLD}/dryrun/stub/tinycudann.py"
do_mv "${KIT_OLD}/.upstream"                     "${KIT_OLD}/.upstream"
[ "${DRY}" -eq 0 ] && rmdir "${KIT_OLD}/dryrun/stub" "${KIT_OLD}/dryrun" 2>/dev/null
do_mv "${KIT_OLD}" "${KIT_NEW}"

echo "── 2. 修正所有文件里的相互引用"
# 按字符串长度倒序替换，否则 dryrun 会先吃掉 run.py 里的子串
SED_ARGS=()
while IFS='|' read -r o n; do [ -z "$o" ] && continue; SED_ARGS+=("${o}|${n}"); done <<< "${MAP}"
while IFS='|' read -r o n; do [ -z "$o" ] && continue; SED_ARGS+=("${o}|docs/${n}"); done <<< "${DOCS}"
SED_ARGS+=("run.py|run.py" "fake_data.py|fake_data.py"
           "dryrun|dryrun" ".upstream|.upstream" "${KIT_OLD}|${KIT_NEW}")
mapfile -t SORTED < <(printf '%s\n' "${SED_ARGS[@]}" | awk -F'|' '{print length($1)"\t"$0}' | sort -rn | cut -f2-)
if [ "${DRY}" -eq 1 ]; then
  echo "  [预演] 会按长度倒序替换 ${#SORTED[@]} 组名称"
else
  mapfile -t FILES < <(find "${KIT_NEW}" docs -type f \( -name '*.sh' -o -name '*.py' -o -name '*.md' -o -name 'gitignore' \) 2>/dev/null)
  for pair in "${SORTED[@]}"; do
    o="${pair%%|*}"; n="${pair#*|}"
    sed -i "s|${o//./\\.}|${n}|g" "${FILES[@]}" 2>/dev/null
  done
  echo "  已处理 ${#FILES[@]} 个文件"
fi

[ "${DRY}" -eq 1 ] && { echo; echo "预演结束。去掉 --dry 真正执行。"; exit 0; }

echo "── 2b. 产物名（只改路径，不动日志里 Gate A/B 的说法 —— 那是项目既有术语）"
if [ "${DRY}" -eq 0 ]; then
  mapfile -t T < <(find "${KIT_NEW}" -type f \( -name '*.sh' -o -name '*.py' -o -name '*.md' -o -name 'gitignore' \))
  sed -i 's|gate_b_acceptance\.txt|acceptance.txt|g'          "${T[@]}"
  sed -i 's|runtime_signature\.json|sysinfo.json|g'           "${T[@]}"
  sed -i 's|run_all_gates\.log|gates.log|g'                   "${T[@]}"
  sed -i 's|/gate_a_\${STAMP}|/doctor_${STAMP}|g'             "${KIT_NEW}/doctor.sh"
  sed -i "s|'gate_a_\*'|'doctor_*'|g"                         "${KIT_NEW}/gates.sh"
  sed -i 's|/gate_a_\*/|/doctor_*/|g'                         "${KIT_NEW}/gates.sh"
  sed -i 's|gate_b_smoke\${EPOCHS}_\${ARM}_seed\${SEED}_\${STAMP}|smoke_${ARM}_${STAMP}|g' "${KIT_NEW}/smoke.sh"
  sed -i 's|"gate_b_\*_\${arm}_\*"|"smoke_${arm}_*"|g'      "${KIT_NEW}/gates.sh"
  sed -i 's|/gate_b_\*/|/smoke_*/|g'                          "${KIT_NEW}/gates.sh"
  sed -i 's|smoke_|smoke_|g'                          "${T[@]}"
  sed -i 's|gate_b_\*|smoke_*|g; s|gate_b_\.|smoke_.|g'        "${T[@]}"
  # 最后两处把硬件写进名字的地方
  sed -i 's|^echo "env_5070:|echo "env:|' "${KIT_NEW}/env.sh"
  sed -i 's|runs|runs|g' "${T[@]}"
  # --tag 会进 run_info.json 和输出目录名，属于「名字」；其余 5070 是散文和真实 venv 路径，保留
  sed -i 's|--tag "smoke\${EPOCHS}_5070_\${ARM}"|--tag "smoke_${ARM}"|' "${KIT_NEW}/smoke.sh"
  sed -i 's|--tag "gateb_\${BEST}"|--tag "smoke_${BEST}"|' "${KIT_NEW}/autopilot.sh"
  echo "  已改：runs/ -> runs/（产物根目录不该带显卡型号；旧的 ~/runs 留在原地）"
  echo "  已改：gate_a_<ts>/ -> doctor_<ts>/，smoke_U0_seed0_<ts>/ -> smoke_U0_<ts>/"
  echo "        acceptance.txt -> acceptance.txt，sysinfo.json -> sysinfo.json"
else
  echo "  [预演] 会清理产物目录名与输出文件名"
fi

echo "── 3. 自检"
BAD=0
for f in $(find "${KIT_NEW}" -name '*.sh'); do bash -n "$f" 2>/dev/null || { echo "  ✗ bash 语法: $f"; BAD=1; }; done
for f in $(find "${KIT_NEW}" -name '*.py'); do python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" "$f" 2>/dev/null || { echo "  ✗ python 语法: $f"; BAD=1; }; done
# 只查「文件/路径」形态的旧名。日志里 "Gate A 通过" 这类措辞是项目术语，保留。
# docs/ 里的带日期报告不参与检查：它们记录的是当时真实的目录名，改了就是篡改记录
LEFT=$(grep -rlE 'tools|env_5070\.sh|check_smoke|probe_tcnn_5070|preflight_imports|runtime_signature|dryrun|projection_audit\.(py|csv|json)|run_all_gates|autopilot_gate_b|run_with_retry|fix_missing_deps|gate_[ab]_[a-z]*[/*.]' "${KIT_NEW}" 2>/dev/null)
[ -n "${LEFT}" ] && { echo "  ✗ 仍有旧名残留:"; echo "${LEFT}" | sed 's/^/      /'; BAD=1; }
[ "${BAD}" -eq 0 ] && echo "  ✓ 语法全过，无旧名残留"
echo
echo "新的目录结构:"; find "${KIT_NEW}" docs -type f 2>/dev/null | sort | sed 's/^/  /'
exit "${BAD}"
