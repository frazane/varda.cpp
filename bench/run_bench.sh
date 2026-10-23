#!/bin/bash
# varda.cpp benchmark driver: one A100, one job, hook-free, warm, input on the device.
#   source tools/env.sh
#   sbatch $VARDA_SLURM_GPU --time=01:00:00 -o $VARDA_SCRATCH/jobs/bench/slurm-%j.out bench/run_bench.sh varda   (or dev)
# Env: VARDA_SCRATCH, VARDA_CKPT / VARDA_DEV_CKPT, REF_ROOT, REF_GOLDEN / REF_GOLDEN_DEV, REF_VENV / REF_VENV_DEV
#      (tools/env.sh). Optional: WARMUP (5) ITERS (20)  STEPS ("ref varda")  BIN (dir with varda-bench)
#      VARDA_ROOT (repo)  FP32_ITERS (default = ITERS)
# Outputs: $VARDA_SCRATCH/jobs/bench/<jobid>/{meta.json,results/*,logs/*,exit_codes.txt,summary.md}
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH -J varda-bench
set -uo pipefail

M=${1:?usage: run_bench.sh varda|dev}
# sbatch runs a copy of this script: the repository is the submit directory unless VARDA_ROOT says otherwise
VARDA_ROOT=${VARDA_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}
[[ -n ${VARDA_SCRATCH:-} && -n ${REF_ROOT:-} ]] || source "$VARDA_ROOT/tools/env.sh" || exit 2  # reference data: read-only
BIN=${BIN:-$VARDA_ROOT/build-cuda}
WARMUP=${WARMUP:-5}; ITERS=${ITERS:-20}; FP32_ITERS=${FP32_ITERS:-$ITERS}
STEPS=${STEPS:-"ref varda"}
JID=${SLURM_JOB_ID:-local$$}
J=$VARDA_SCRATCH/jobs/bench/$JID
CACHE=$VARDA_SCRATCH/jobs/bench/cache/$M    # warm triton cache, ours only
mkdir -p "$J/results" "$J/logs" "$J/src" "$CACHE"
exec >"$J/logs/driver.log" 2>&1

case $M in
  varda)
    CKPT=${VARDA_CKPT:?set VARDA_CKPT (tools/env.local.sh)}
    REFPY=${REF_VENV:?set REF_VENV (tools/env.local.sh)}/bin/python
    INPUT=${REF_GOLDEN:?set REF_GOLDEN (tools/env.local.sh)}/cpu_float32_pyg/input.pt ;;
  dev)
    CKPT=${VARDA_DEV_CKPT:?set VARDA_DEV_CKPT (tools/env.local.sh)}
    REFPY=${REF_VENV_DEV:?set REF_VENV_DEV (tools/env.local.sh)}/bin/python
    INPUT=${REF_GOLDEN_DEV:?set REF_GOLDEN_DEV (tools/env.local.sh)}/cpu_float32_pyg/input.pt ;;
  *) echo "unknown model $M"; exit 2 ;;
esac
V_MODEL=$VARDA_SCRATCH/converted/$M/model.safetensors
V_INPUT=$VARDA_SCRATCH/golden/$M/cpu-fp32-pyg.safetensors

# Job-private copies of the reference timing driver (read-only, may change under us); never write there.
cp -r "$REF_ROOT/bench" "$REF_ROOT/reference" "$J/src/"
find "$J/src" -name __pycache__ -prune -exec rm -rf {} +
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=$CACHE/triton-cache
if [ -x "$BIN/varda-bench" ]; then mkdir -p "$J/bin" && cp "$BIN/varda-bench" "$J/bin/"; fi

# metadata
{
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1 | \
    awk -F', ' -v n="$(hostname)" -v j="$JID" -v m="$M" \
    '{printf "{\"gpu\":\"%s\",\"driver\":\"%s\",\"mem_total\":\"%s\",\"node\":\"%s\",\"job_id\":\"%s\",\"model\":\"%s\"}\n",$1,$2,$3,n,j,m}'
} > "$J/meta.json"
cat "$J/meta.json"
echo "model=$M warmup=$WARMUP iters=$ITERS steps=$STEPS cache=$CACHE"

: > "$J/exit_codes.txt"
run() {  # run NAME CMD... : log to logs/NAME.log, record exit code
  local name=$1; shift
  echo "[$(date +%T)] $name: $*"
  "$@" >"$J/logs/$name.log" 2>&1; local rc=$?
  echo "$name $rc" >> "$J/exit_codes.txt"; echo "[$(date +%T)] $name rc=$rc"
}

# results/<model>.<impl>.<precision>.jsonl  (summarize.py takes impl and precision from the file name)
for p in fp16 fp32; do
  it=$ITERS; [ $p = fp32 ] && it=$FP32_ITERS
  ac=none; [ $p = fp16 ] && ac=fp16
  W="--warmup $WARMUP --repeats $it"
  case " $STEPS " in *" ref "*)
    # Original anemoi model, triton backend, hook-free, CUDA-event timer (bench_reference.py is
    # read-only: it does not write a manifest).
    run ref-$p "$REFPY" "$J/src/bench/bench_reference.py" --backend triton --autocast $ac $W \
        --ckpt "$CKPT" --input "$INPUT" --label "original-$p" --json "$J/results/$M.original.$p.jsonl" ;;
  esac
  case " $STEPS " in *" varda "*)
    if [ -x "$J/bin/varda-bench" ]; then
      # >>> adapt this one line to varda-bench's CLI <<<
      run varda-$p "$J/bin/varda-bench" --model "$V_MODEL" --input "$V_INPUT" --device cuda --precision $p --warmup $WARMUP --iters $it --json "$J/results/$M.varda.$p.json"
    else
      echo "varda-bench not found at $BIN/varda-bench: skipping varda.cpp ($p)"
      echo "varda-$p skipped" >> "$J/exit_codes.txt"
    fi ;;
  esac
done

python3 "$VARDA_ROOT/bench/summarize.py" "$J" > "$J/summary.md" 2>"$J/logs/summarize.err"
cat "$J/summary.md"
