#!/usr/bin/env bash
# ON the instance: Phase 17 stages 3a + 3 on the cloud GPU (user's go 2026-11-08), started by
# scripts/cloud/chain_p17.sh when the p15 session ends (same instance, no relaunch).
#
#   3a  teacher pilot: the same PILOT_N stage-3 pool problems translated by the 14B and by each
#       teacher (vLLM, product prompt + retry loop), judged by the problems' own tests
#       → data/reports/cloud/p17/pilot_<model>.json; a teacher replaces the 14B only if its
#       verified rate is ≥ 5 points higher (PHASE_17: "use the 32B only if it clearly wins")
#   3   stage-3 generation with the chosen model: up to S3_MAX problems or S3_HOURS hours
#       (training/gen_xlang_pairs.py generate --backend vllm), then `pairs`
#   4+  scripts/cloud/session_p17_stage4.sh if present (shared adapter, eval, DPO rollout)
#
# Restartable: generations and verdicts are cached per model; finished steps are skipped.
set -uo pipefail
cd ~/rebirth
export REBIRTH_TRACKING=0 HF_HUB_OFFLINE=1 REBIRTH_EMBEDDER=hash
D=logs/cloud/p17
R=data/reports/cloud/p17
X=data/datasets/xlang/stage3
mkdir -p "$D" "$R"
B14=Qwen/Qwen2.5-Coder-14B-Instruct
B7=Qwen/Qwen2.5-Coder-7B-Instruct
TEACHERS="${TEACHERS:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 Qwen/Qwen3.6-27B-FP8}"
PILOT_N="${PILOT_N:-40}"
S3_MAX="${S3_MAX:-2000}"
S3_HOURS="${S3_HOURS:-9}"
MARGIN="${MARGIN:-5}"   # verified-rate points a teacher must gain over the 14B
source scripts/cloud/hub.sh   # tag, cached, drop, fetch, free_gb, extra_for

note() { echo "{\"t\":\"$(date -Is)\",$1}" | tee -a "$D/status.jsonl"; }
unserve() { docker rm -f rebirth-vllm >/dev/null 2>&1 || true; }
serve() {  # serve <model>: 8K context for program translations (prompt + retry context + output);
  # READY_MIN=30: a teacher's first load from disk takes ~12 min (a 10-min wait dropped one in the first run)
  unserve
  if ! MAX_MODEL_LEN=8192 READY_MIN=30 VLLM_EXTRA="$(extra_for "$1")" bash scripts/cloud/vllm_serve.sh "$1"; then
    note "\"event\":\"vllm_failed\",\"model\":\"$1\""; docker logs --tail 60 rebirth-vllm 2>&1 | tail -n 60; unserve; return 1
  fi
}
gen() {  # gen <model> <generate args…>
  local m=$1; shift
  python -u training/gen_xlang_pairs.py generate --backend vllm --model "$m" --max-model-len 8192 \
    --concurrency 48 --pipelines 2 --workers 3 --chunk 25 "$@"
}

note '"event":"p17_start"'
# disk: the 7B base was only needed for the S1 baselines (a copy stays at home)
cached "$B7" && { drop "$B7"; note "\"event\":\"dropped_7b\",\"free_gb\":$(free_gb)"; }
# resumable trainer checkpoints of finished adapters (~2.4 GB each; never synced home or to S3)
for d in data/models/lora_*/; do [ -f "$d/adapter/adapter_model.safetensors" ] && rm -rf "$d/runs"; done
[ -f "$D/start_epoch" ] || date +%s > "$D/start_epoch"   # session start (time budget for the last steps)
note "\"event\":\"disk\",\"free_gb\":$(free_gb)"

# ── sandboxes (built during p15 by /tmp/build_sandboxes.sh; rebuild whatever is missing) ────────
for _ in $(seq 1 60); do grep -q "sandbox build done" "$D/sandbox_build.log" 2>/dev/null && break; sleep 30; done
missing=0
for l in python js_ts go java cpp rust csharp swift; do
  img="rebirth-ai/${l//_/-}-sandbox:latest"
  docker image inspect "$img" >/dev/null 2>&1 || docker build -q -t "$img" "docker/${l}_sandbox" >> "$D/sandbox_build.log" 2>&1 || missing=$((missing + 1))
done
note "\"event\":\"sandboxes\",\"missing\":$missing"
[ "$missing" -eq 0 ] || { note '"event":"session_end","why":"sandbox images missing"'; exit 1; }

# ── 3a teacher pilot ───────────────────────────────────────────────────────────────────────────
note "\"event\":\"pilot_start\",\"n\":$PILOT_N"
for m in "$B14" $TEACHERS; do
  t=$(tag "$m")
  [ -f "$R/pilot_$t.json" ] && continue
  if [ "$m" != "$B14" ]; then
    for o in $TEACHERS; do [ "$o" != "$m" ] && drop "$o"; done   # one teacher on disk at a time
    fetch "$m" || continue
  fi
  serve "$m" || continue
  gen "$m" --pilot "$PILOT_N" --cache-dir "$X/pilot/$t" --report "$R/pilot_$t.json" > "$D/pilot_$t.log" 2>&1
  note "\"event\":\"pilot_model_end\",\"model\":\"$m\",\"exit\":$?"
  unserve
done
python - "$R" "$B14" "$MARGIN" $TEACHERS <<'PY'
import json, sys
from pathlib import Path
R, base, margin, teachers = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4:]
tag = lambda m: m.split("/")[-1].lower().replace(".", "_")
rows = {}
for m in [base, *teachers]:
    f = R / f"pilot_{tag(m)}.json"
    if f.is_file():
        s = json.loads(f.read_text())
        rows[m] = {k: s[k] for k in ("translations", "first_try_pct", "verified_pct", "generations", "wall_hours")}
        rows[m]["by_target_verified"] = {t: v["verified"] for t, v in s["by_target"].items()}
base_pct = rows.get(base, {}).get("verified_pct", 0.0)
best = max((m for m in teachers if m in rows), key=lambda m: rows[m]["verified_pct"], default=None)
choice = best if best and rows[best]["verified_pct"] >= base_pct + margin else base
out = {"rule": f"teacher only if verified rate >= 14B + {margin} points (same {len(rows)} models, same problems)",
       "models": rows, "choice": choice}
(R / "pilot_choice.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
PY
CHOICE=$(python -c "import json; print(json.load(open('$R/pilot_choice.json'))['choice'])" 2>/dev/null || echo "$B14")
note "\"event\":\"pilot_end\",\"choice\":\"$CHOICE\""

# ── 3 generation with the chosen model ─────────────────────────────────────────────────────────
ct=$(tag "$CHOICE")
if [ ! -f "$X/pilot/$ct/merged" ] && [ -f "$X/pilot/$ct/verdicts.jsonl" ]; then
  cat "$X/pilot/$ct/gens.jsonl" >> "$X/gens.jsonl"; cat "$X/pilot/$ct/verdicts.jsonl" >> "$X/verdicts.jsonl"
  touch "$X/pilot/$ct/merged"   # the pilot problems count as done for the main run
fi
if [ ! -f "$R/stage3_generate.json" ]; then
  [ "$CHOICE" = "$B14" ] || { for o in $TEACHERS; do [ "$o" != "$CHOICE" ] && drop "$o"; done; fetch "$CHOICE"; }
  if serve "$CHOICE"; then
    note "\"event\":\"stage3_start\",\"model\":\"$CHOICE\",\"max_problems\":$S3_MAX,\"hours\":$S3_HOURS"
    gen "$CHOICE" --max-problems "$S3_MAX" --deadline-hours "$S3_HOURS" --report "$R/stage3_generate.json" \
      >> "$D/stage3_generate.log" 2>&1
    note "\"event\":\"stage3_end\",\"exit\":$?"
  fi
  unserve
fi
python -u training/gen_xlang_pairs.py pairs --workers 4 > "$D/stage3_pairs.log" 2>&1
note "\"event\":\"pairs_end\",\"exit\":$?,\"rows\":$(wc -l < "$X/pairs.jsonl" 2>/dev/null || echo 0)"
[ -n "${REBIRTH_S3_BUCKET:-}" ] && aws s3 sync "$X" "s3://$REBIRTH_S3_BUCKET/results/xlang_stage3" --only-show-errors

# ── 4+ (shared adapter, its eval, DPO rollout): own script, synced up before this point ────────
for i in $(seq 1 120); do  # not synced up yet: wait up to 2 h (heartbeat keeps the watchdog from stopping)
  [ -f scripts/cloud/session_p17_stage4.sh ] && break
  [ $((i % 10)) -eq 1 ] && note '"event":"waiting_for_stage4_script"'
  sleep 60
done
if [ -f scripts/cloud/session_p17_stage4.sh ]; then
  bash scripts/cloud/session_p17_stage4.sh
  note "\"event\":\"stage4_script_end\",\"exit\":$?"
fi
note '"event":"session_end","why":"all stages ran"'
