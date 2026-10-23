#!/bin/bash
# sbatch wrapper for run_reference_realdata.py. Copies the driver into a job-private directory first
# (CLAUDE.md: a running job never executes files someone may edit), then runs one or more configurations
# sequentially; logs and exit codes go to $VARDA_SCRATCH/jobs/<name>/.
#
#   source tools/env.sh   # from the repository root (the submit directory is the repository)
#   NAME=realdata-ref-gpu sbatch $VARDA_SLURM_GPU --time=02:00:00 \
#       -o $VARDA_SCRATCH/jobs/realdata-ref-gpu/slurm-%j.out python/reference/realdata/job_reference.sh \
#       "--device cuda --precision checkpoint --out $R/ref-fp16" "--device cuda --precision 32 --out $R/ref-fp32"
#
# Each positional argument is one run's extra arguments (DATE defaults to 2026-10-22T15).
set -u
VARDA_SCRATCH=${VARDA_SCRATCH:?set VARDA_SCRATCH (source tools/env.sh)}
NAME=${NAME:?set NAME}
DATE=${DATE:-2026-10-22T15}
REPO=${REPO:-${SLURM_SUBMIT_DIR:?set REPO to the repository}}
JOB=$VARDA_SCRATCH/jobs/$NAME
PY=$VARDA_SCRATCH/ref-oper/.venv/bin/python
mkdir -p "$JOB"
CODE=$JOB/code-${SLURM_JOB_ID:-local}
mkdir -p "$CODE/python/reference/realdata"
cp "$REPO/python/safetensors_io.py" "$CODE/python/"
cp "$REPO"/python/reference/realdata/*.py "$CODE/python/reference/realdata/"
i=0
for run in "$@"; do
    i=$((i + 1))
    log=$JOB/run$i-${SLURM_JOB_ID:-local}.log
    echo "[$(date -u +%FT%TZ)] run $i: $run" | tee -a "$JOB/runs.txt"
    # shellcheck disable=SC2086
    $PY -O -I "$CODE/python/reference/realdata/run_reference_realdata.py" --date "$DATE" $run > "$log" 2>&1
    rc=$?
    echo "$rc" > "$JOB/run$i-${SLURM_JOB_ID:-local}.exit"
    echo "[$(date -u +%FT%TZ)] run $i exit $rc" | tee -a "$JOB/runs.txt"
done
