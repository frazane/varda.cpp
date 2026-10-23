# Machine-specific settings for varda.cpp. Copy to tools/env.local.sh (untracked) and fill in;
# tools/env.sh sources it. Leave a toolchain variable empty to use what is on PATH.

# Toolchain. VARDA_GCC: a gcc >= 11 install prefix (sets CC, CXX, CUDAHOSTCXX).
export VARDA_GCC=
# CUDA toolkit prefix (nvcc, cuBLAS); empty: build the CPU backend only, or use nvcc on PATH.
export CUDA_HOME=
# OpenBLAS install prefix (cblas.h, libopenblas).
export OPENBLAS_HOME=

# Writable directory for large artefacts: converted models, exported golden, job logs, assets bundle,
# input cache. Never inside the repository.
export VARDA_SCRATCH=/path/to/varda-scratch

# The Hugging Face Varda forecaster checkpoint (MeteoSwiss/Varda-single-1.0,
# varda-forecaster-det-sgm/model.ckpt). Read-only.
export VARDA_CKPT=$VARDA_SCRATCH/hf/varda-forecaster-det-sgm/model.ckpt
# The dev model checkpoint: a smaller forecaster of the same architecture, for fast iteration. Read-only.
export VARDA_DEV_CKPT=/path/to/dev-model.ckpt

# Reference data made with anemoi (read-only, never written by this project).
# REF_ROOT: reference checkout (anemoi reference drivers, timing driver, venv recipe).
export REF_ROOT=/path/to/reference
# REF_SCRATCH: reference data (golden activations, reference rollouts, venvs).
export REF_SCRATCH=/path/to/reference-data
# Golden activations of Varda, one directory per tag (cpu-fp32-pyg, cuda-fp32-triton, ...).
export REF_GOLDEN=$REF_SCRATCH/golden/varda
# Golden activations of the dev model, same tags.
export REF_GOLDEN_DEV=$REF_SCRATCH/golden/dev
# Python venv with the original anemoi stack for Varda (bench/run_bench.sh, reference drivers).
export REF_VENV=$REF_SCRATCH/venv-ref-varda
# Same for the dev model.
export REF_VENV_DEV=$REF_ROOT/.venv-ref
# The package lock REF_VENV was built from (python/reference/run_reference.py recipe).
export REF_LOCK=$REF_ROOT/reference/requirements-ref-varda.lock.txt

# ecCodes for the GRIB layer (-DVARDA_GRIB=ON): prefix with eccodeslib/{include,lib64}, and the COSMO
# definitions directory that KENDA-CH1 GRIB needs.
export ECCODES_HOME=/path/to/eccodes
export VARDA_COSMO_DEFINITIONS=/path/to/eccodes-cosmo-resources/definitions

# Slurm arguments (sbatch/srun) for one GPU job and one CPU job; used by ./varda and the job scripts.
export VARDA_SLURM_GPU="-p GPU_PARTITION -A ACCOUNT --gres=gpu:1 -c 32 --mem=200G --time=00:30:00"
export VARDA_SLURM_CPU="-p CPU_PARTITION -A ACCOUNT -c 64 --mem=200G --time=01:00:00"
