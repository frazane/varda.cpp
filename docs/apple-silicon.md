# Apple Silicon (macOS arm64)

The existing CPU fp32 backend runs natively on Apple Silicon using Apple Clang,
OpenMP and OpenBLAS. The optional native Metal/MPS backend runs fp32 and mixed-fp16 inference on
the Apple GPU; see [metal.md](metal.md). CUDA remains NVIDIA-only. CPU arithmetic
and numerical tolerances are unchanged.

## CPU build

Install Xcode Command Line Tools and the build dependencies:

```sh
brew install cmake libomp openblas
cmake -S . -B build-apple -DVARDA_CUDA=OFF -DVARDA_METAL=OFF -DVARDA_GRIB=OFF \
  -DCMAKE_PREFIX_PATH="$(brew --prefix libomp);$(brew --prefix openblas)"
cmake --build build-apple -j 4
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 ctest --test-dir build-apple --output-on-failure
```

The prefix list lets CMake discover Homebrew's keg-only OpenMP and OpenBLAS
installations. The build prefers OpenBLAS's `cblas.h` over the SDK's Accelerate
header, and does not pass x86 AVX flags to the ARM compiler. `VARDA_NATIVE=ON`
remains optional for tuning to the build host.

## GRIB build and synthetic verification

The small test environment uses versions from the existing reference locks. It
does not need torch or Anemoi:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r python/tests/requirements-cpu.txt
export ECCODES_HOME="$(.venv/bin/python -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cmake -S . -B build-apple -DVARDA_CUDA=OFF -DVARDA_GRIB=ON \
  -DCMAKE_PREFIX_PATH="$(brew --prefix libomp);$(brew --prefix openblas)"
cmake --build build-apple -j 4
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 ctest --test-dir build-apple --output-on-failure
BIN="$PWD/build-apple" SYN_DIR="$PWD/local/synthetic" \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 bash tests/run_synthetic.sh cpu
```

The macOS ecCodes wheel uses `eccodeslib/lib` instead of Linux's
`eccodeslib/lib64`. Both layouts are supported. Homebrew ecCodes can also be
linked, but `test_io_link` deliberately checks the reference version, 2.47.3;
using a different version fails that check and does not establish GRIB parity.

The synthetic suite checks all six configurations at 1 and 8 threads, plus a
different chunk size (18 checks), against an independent float64 reference with
the existing tolerances. Tests that need model checkpoints, recorded forcings,
GRIB templates or real-data dumps still skip those parts when fixtures are absent.
The CPU portability workflow runs these checks on macOS arm64 and Linux x86_64.

## Full Varda forecasts

Use `--device cpu --precision fp32`. The full model's measured CPU footprint on
Linux is approximately 25 GB before allowing for other applications and forecast
I/O; a 16 GB Mac is not a practical target. Synthetic parity is not a full-model
or real-data forecast qualification on macOS. In particular, macOS libm and ARM
OpenBLAS may differ in rounding from the Linux reference.

With a prepared assets bundle, the forecast wrapper can run locally without
Slurm (see `quickstart.md` for preparing assets):

```sh
VARDA_ENV=none VARDA_SRUN=0 VARDA_BUILD="$PWD/build-apple" \
  VARDA_SCRATCH=/path/to/scratch VARDA_ASSETS=/path/to/assets \
  ./varda forecast latest --device cpu --precision fp32
```

`varda-run` reports RSS through macOS APIs and converts `ru_maxrss` from bytes;
Linux retains its `/proc` statistics and KiB conversion. Weight prefetch uses
page touching on macOS, while Linux retains `MADV_POPULATE_READ` and its existing
fallback on kernels that do not support it.
