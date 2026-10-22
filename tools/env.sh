# Source this file: toolchain for building and running varda.cpp on this cluster.
#   source tools/env.sh
# The `module` command does not work in non-interactive shells here, so paths are explicit.

_SPACK=/path/to/spack
export VARDA_GCC=$_SPACK/gcc-11.3.0/gcc-12.3.0-kxhppfcdnri5lftlprcqey5t64bwrd2a
export CUDA_HOME=$_SPACK/gcc-12.3.0/cuda-12.3.0-ihzk4gm66og3lpwzblphemuxxk7h7a3e
export OPENBLAS_HOME=$_SPACK/gcc-12.3.0/openblas-0.3.26-qdrpysrppslf56jkquizbvlers2xttal

export PATH=$CUDA_HOME/bin:$VARDA_GCC/bin:$PATH
export CC=$VARDA_GCC/bin/gcc
export CXX=$VARDA_GCC/bin/g++
export CUDAHOSTCXX=$CXX
export CMAKE_PREFIX_PATH=$OPENBLAS_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}

# Large artefacts live on scratch, never in the repository.
export VARDA_SCRATCH=$VARDA_SCRATCH
# Python engine project (read-only here): golden data, reference venvs, its own engine.
export IE_ROOT=$HOME/projects/the Python project
export IE_SCRATCH=$SCRATCH/the Python project
unset _SPACK
