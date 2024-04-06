#!/bin/bash

# Calling `mklCmake.sh`` is equivalent to calling `cmake`.
# This script just ensures the mkl environment variables are available.
# Designed for vscode cmake plugin ("cmake.cmakePath": "${workspaceFolder}/scripts/mklCmake.sh")

source /opt/intel/oneapi/setvars.sh
cmake "$@"
