#!/bin/bash
# varda.cpp benchmark driver: one A100, one job, hook-free, warm, input on the device.
#   sbatch bench/run_bench.sh varda        (or dev)
#   sbatch -p gpu-normal --time=02:00:00 bench/run_bench.sh varda
# Env (optional): WARMUP (5) ITERS (20)  STEPS ("ref engine varda")  BIN (dir with varda-bench)
#                 VARDA_ROOT (repo)  FP32_ITERS (default = ITERS)
# Outputs: $VARDA_SCRATCH/jobs/bench/<jobid>/{meta.json,results/*,logs/*,exit_codes.txt,summary.md}
#          (Slurm stdout: $VARDA_SCRATCH/jobs/bench/slurm-<jobid>.out)
#SBATCH -A $ACCOUNT
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH -J varda-bench
#SBATCH -o $VARDA_SCRATCH/jobs/bench/slurm-%j.out
set -uo pipefail

M=${1:?usage: run_bench.sh varda|dev}
VARDA_ROOT=${VARDA_ROOT:-$VARDA_ROOT}
VARDA_SCRATCH=${VARDA_SCRATCH:-$VARDA_SCRATCH}
IE_ROOT=${IE_ROOT:-$HOME/projects/the Python project}
IE_SCRATCH=${IE_SCRATCH:-$SCRATCH/the Python project}
export SCRATCH=${SCRATCH:-$SCRATCH}       # used by $IE_ROOT/tools/run.sh
BIN=${BIN:-$VARDA_ROOT/build-cuda}
WARMUP=${WARMUP:-5}; ITERS=${ITERS:-20}; FP32_ITERS=${FP32_ITERS:-$ITERS}
STEPS=${STEPS:-"ref engine varda"}
JID=${SLURM_JOB_ID:-local$$}
J=$VARDA_SCRATCH/jobs/bench/$JID
CACHE=$VARDA_SCRATCH/jobs/bench/cache/$M    # warm inductor/triton cache, ours only
mkdir -p "$J/results" "$J/logs" "$J/src" "$CACHE"
exec >"$J/logs/driver.log" 2>&1

case $M in
  varda)
    CKPT=$VARDA_CKPT
    REFPY=$IE_SCRATCH/venv-ref-varda/bin/python
    INPUT=$IE_SCRATCH/golden/varda/cpu_float32_pyg/input.pt ;;
  dev)
    CKPT=$VARDA_DEV_CKPT
    REFPY=$IE_ROOT/.venv-ref/bin/python
    INPUT=$IE_SCRATCH/golden/cpu_float32_pyg/input.pt ;;
  *) echo "unknown model $M"; exit 2 ;;
esac
ENGINE_MODEL_DIR=$IE_SCRATCH/converted/$M
V_MODEL=$VARDA_SCRATCH/converted/$M/model.safetensors
V_INPUT=$VARDA_SCRATCH/golden/$M/cpu-fp32-pyg.safetensors

# Job-private copies of everything we run (the Python project may change under us); never write there.
cp -r "$IE_ROOT/engine" "$IE_ROOT/bench" "$IE_ROOT/reference" "$J/src/"
find "$J/src" -name __pycache__ -prune -exec rm -rf {} +
export PYTHONDONTWRITEBYTECODE=1
export ENGINE_RUNS=$CACHE
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor-cache TRITON_CACHE_DIR=$CACHE/triton-cache
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
    # Original anemoi model, triton backend, hook-free (same protocol/CUDA-event timer as the Python
    # project's reference/time_gpu.py, but read-only: bench_reference.py does not write a manifest).
    run ref-$p "$REFPY" "$J/src/bench/bench_reference.py" --backend triton --autocast $ac $W \
        --ckpt "$CKPT" --input "$INPUT" --label "original-$p" --json "$J/results/$M.original.$p.jsonl" ;;
  esac
  case " $STEPS " in *" engine "*)
    # Python engine fast path (compile + CUDA graph): image python via the project's run.sh.
    run engine-$p "$IE_ROOT/tools/run.sh" "$J/src/bench/bench.py" --impl fast --precision $p $W \
        --model-dir "$ENGINE_MODEL_DIR" --input "$INPUT" --label "engine-fast-$p" \
        --json "$J/results/$M.pyengine.$p.jsonl" ;;
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
