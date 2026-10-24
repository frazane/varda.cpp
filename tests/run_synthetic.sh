#!/usr/bin/env bash
# Synthetic generality suite (docs/testing.md): tiny models outside the real checkpoints'
# dimension envelope, checked with varda-parity against a float64 numpy reference.
#
#   BIN=build-cuda tests/run_synthetic.sh [cpu] [cuda] [metal] (default: cpu cuda)
#
# Env:
#   BIN       directory holding varda-parity (default: <repo>/build-synth). Required to exist.
#   SYN_DIR   where models and golden files live (default $VARDA_SCRATCH/synthetic). They are
#             (re)generated when missing or when the SHA-256 of python/tests/synthetic.py
#             differs from the one stored in $SYN_DIR/generator.sha256.
#   MODELS    space-separated subset of the configs (default: all of a b c d e f).
#   OUT       directory for logs and per-run JSON reports (default $SYN_DIR/runs/<timestamp>).
#   SKIP_CUDA=1 forces the CUDA part to be skipped (it is also skipped, with a note, when
#             no GPU is visible).
# Runs per model: CPU fp32 at 1 and 8 threads, CPU fp32 with VARDA_CPU_CHUNK_ROWS=37 (8
# threads); CUDA fp32, CUDA fp32 with VARDA_DEC_CHUNK_ROWS=128, CUDA fp16.
# Prints a summary (worst rel_absmax and ratio to the tolerance per run) and exits non-zero
# if any run fails or errors. The varda-parity check includes the hook-inertness check.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$REPO/build-synth}"
PARITY="$BIN/varda-parity"
PY="$REPO/.venv/bin/python"
GEN="$REPO/python/tests/synthetic.py"
[ -n "${SYN_DIR:-}" ] || : "${VARDA_SCRATCH:?set VARDA_SCRATCH (source tools/env.sh) or SYN_DIR}"
SYN_DIR="${SYN_DIR:-$VARDA_SCRATCH/synthetic}"
MODELS="${MODELS:-a b c d e f}"
OUT="${OUT:-$SYN_DIR/runs/$(date +%Y%m%d-%H%M%S)-$$}"
want="${*:-cpu cuda}"

[ -x "$PARITY" ] || { echo "error: $PARITY not found (set BIN to the directory with varda-parity)"; exit 2; }
[ -x "$PY" ] || { echo "error: $PY not found"; exit 2; }
mkdir -p "$SYN_DIR" "$OUT" || exit 2

# ---- generate models if missing or if the generator changed
cur_hash="$("$PY" -I "$GEN" --hash)"
old_hash="$(cat "$SYN_DIR/generator.sha256" 2>/dev/null || true)"
need=0
[ "$cur_hash" = "$old_hash" ] || need=1
for m in $MODELS; do
  [ -f "$SYN_DIR/$m.model.safetensors" ] && [ -f "$SYN_DIR/$m.golden.safetensors" ] || need=1
done
if [ "$need" = 1 ]; then
  echo "generating synthetic models into $SYN_DIR (generator hash ${cur_hash:0:12})"
  # Only the requested subset is regenerated, so the hash file is written only for the full set.
  "$PY" -I "$GEN" --out "$SYN_DIR" --only "$(echo $MODELS | tr ' ' ',')" || { echo "generation failed"; exit 2; }
  if [ "$MODELS" = "a b c d e f" ]; then echo "$cur_hash" > "$SYN_DIR/generator.sha256"; fi
fi
"$PY" -I "$GEN" --check-tolerances "$REPO/tolerances" --only "$(echo $MODELS | tr ' ' ',')" || exit 2

# ---- runs
have_gpu=0
if [ "${SKIP_CUDA:-0}" != 1 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then have_gpu=1; fi
have_metal=0
case " $want " in *" metal "*)
  [ -x "$BIN/varda-metal-info" ] || { echo "error: build has no varda-metal-info (VARDA_METAL=ON)"; exit 2; }
  "$BIN/varda-metal-info"
  probe=$?
  if [ "$probe" = 0 ]; then have_metal=1
  elif [ "$probe" != 77 ]; then echo "error: Metal device probe failed"; exit 2
  fi
;; esac

fail=0
SUMMARY="$OUT/summary.tsv"
: > "$SUMMARY"

# run MODEL LABEL TABLE "ENV=.. ENV=.." parity-args...
run() {
  local m="$1" label="$2" table="$3" envs="$4"; shift 4
  local log="$OUT/$m.$label.log" rep="$OUT/$m.$label.json"
  env $envs "$PARITY" --model "$SYN_DIR/$m.model.safetensors" \
      --check "$SYN_DIR/$m.golden.safetensors=$table" \
      --tolerances "$REPO/tolerances/synthetic-$m.json" --report "$rep" "$@" > "$log" 2>&1
  local rc=$?
  local line
  line="$("$PY" -I - "$rep" "$m" "$label" "$rc" <<'EOF'
import json, sys
rep, m, label, rc = sys.argv[1:5]
try:
    r = json.load(open(rep))
except Exception:
    print(f"{m}\t{label}\tERROR\t-\t-\t-\tno report (see log)"); sys.exit(0)
if "exception" in r:
    print(f"{m}\t{label}\tERROR\t-\t-\t-\t{r['exception'][:80]}"); sys.exit(0)
c = r["checks"][0]
worst = max(((v["rel_absmax"], k) for k, v in c["keys"].items()), default=(0, "-"))
bad = [k for k, v in c["keys"].items() if not v["ok"]]
st = "PASS" if int(rc) == 0 else "FAIL"
extra = ("failed keys: " + ",".join(bad[:4]) + ("..." if len(bad) > 4 else "")) if bad else (c["errors"][0][:60] if c["errors"] else "")
print(f"{m}\t{label}\t{st}\t{worst[0]:.2e}\t{worst[1]}\t{c['worst_ratio']:.3f}\t{extra}")
EOF
)"
  echo "$line" >> "$SUMMARY"
  case "$line" in *$'\tPASS\t'*) ;; *) fail=1 ;; esac
  echo "  $line"
}

echo "BIN=$BIN OUT=$OUT"
for m in $MODELS; do
  case " $want " in *" cpu "*)
    run "$m" cpu-fp32-1thr   synthetic-fp32 "" --device cpu --precision fp32 --threads 1
    run "$m" cpu-fp32-8thr   synthetic-fp32 "" --device cpu --precision fp32 --threads 8
    run "$m" cpu-fp32-chunk37 synthetic-fp32 "VARDA_CPU_CHUNK_ROWS=37" --device cpu --precision fp32 --threads 8
  ;; esac
  case " $want " in *" cuda "*)
    if [ "$have_gpu" = 1 ]; then
      run "$m" cuda-fp32        synthetic-fp32 "" --device cuda --precision fp32
      run "$m" cuda-fp32-chunk128 synthetic-fp32 "VARDA_DEC_CHUNK_ROWS=128" --device cuda --precision fp32
      run "$m" cuda-fp16        synthetic-fp16 "" --device cuda --precision fp16
    fi
  ;; esac
  case " $want " in *" metal "*)
    if [ "$have_metal" = 1 ]; then
      run "$m" metal-fp32 synthetic-fp32 "" --device metal --precision fp32
      run "$m" metal-fp32-chunk37 synthetic-fp32 "VARDA_METAL_CHUNK_ROWS=37" --device metal --precision fp32
    fi
  ;; esac
done

echo
echo "model  run                 status  worst rel_absmax  (key)            ratio-to-tol"
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
case " $want " in *" cuda "*) [ "$have_gpu" = 1 ] || echo "NOTE: no GPU visible (or SKIP_CUDA=1): the CUDA runs were skipped, this is NOT a full pass." ;; esac
case " $want " in *" metal "*)
  if [ "$have_metal" = 0 ]; then
    echo "NOTE: no Metal GPU: Metal checks were not run (exit 77), this is NOT a GPU pass."
    [ "$fail" = 0 ] && exit 77
  fi
;; esac
if [ "$fail" = 0 ]; then echo "synthetic suite: PASS"; else echo "synthetic suite: FAIL (logs in $OUT)"; fi
exit "$fail"
