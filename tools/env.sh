# Source this file: toolchain and paths for building and running varda.cpp.
#   source tools/env.sh
# Machine-specific values live in tools/env.local.sh (untracked; template: tools/env.local.example.sh),
# sourced first; this file derives the rest.

_VARDA_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ ! -f "$_VARDA_ENV_DIR/env.local.sh" ]; then
    echo "tools/env.sh: $_VARDA_ENV_DIR/env.local.sh is missing: copy tools/env.local.example.sh to" \
         "tools/env.local.sh and fill in this machine's paths" >&2
    unset _VARDA_ENV_DIR
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
. "$_VARDA_ENV_DIR/env.local.sh"
unset _VARDA_ENV_DIR

if [ -n "${CUDA_HOME:-}" ]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
if [ -n "${VARDA_GCC:-}" ]; then
    export PATH="$VARDA_GCC/bin:$PATH"
    export CC="$VARDA_GCC/bin/gcc"
    export CXX="$VARDA_GCC/bin/g++"
    export CUDAHOSTCXX="$CXX"
fi
if [ -n "${OPENBLAS_HOME:-}" ]; then
    export CMAKE_PREFIX_PATH="$OPENBLAS_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
fi
