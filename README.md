
<img src="https://github.com/AlabamaASRL/asset_asrl/assets/40646929/ebd03b27-df85-41bb-9cd0-6bfa06c14f26" width="300" height="300" align="center">

# ASSET: Astrodynamics Software and Science Enabling Toolkit

ASSET (Astrodynamics Software and Science Enabling Toolkit) is a modular, extensible library for trajectory design and optimal control.
It uses a custom implementation of vector math formalisms to enable rapid implementation of dynamical systems and automatic differentiation.
The phase object is the core of the optimal control functionality, and by linking multiple phases together, the user can construct scenarios of arbitrary complexity.
A newly developed high performance interior-point optimizer (PSIOPT) is included with the library, which enables quick turnaround from concept to solution.

Development funded by NASA under Grant No. 80NSSC19K1643.

## Download
-----

You can obtain precompiled binaries from pypi using pip.

```
pip install asset-asrl
```

## Platform Support
-----

ASSET supports the following platforms:
- **Linux** (x86-64): Full support with Intel MKL
- **macOS** (x86-64 and ARM64): Full support with Intel MKL
- **Windows** (x86-64): Full support with Intel MKL
- **Windows ARM64**: Support with OpenBLAS (builds from source)

### Building from Source on Windows ARM64

On Windows ARM64 systems, Intel's oneAPI Math Kernel Library (MKL) is not available. When building from source on these systems, ASSET will automatically use **OpenBLAS** for BLAS/LAPACK operations and **SuiteSparse LDL** for sparse linear algebra.

**Prerequisites:**
1. **Install ARM64 Python** (Critical - must match architecture):
   - Download ARM64 Python from: https://www.python.org/downloads/windows/
   - Look for "Windows installer (ARM64)" download
   - During installation, check "Include development headers (Python.h)"
   - Verify: `python -c "import platform; print(platform.machine())"` should output `ARM64`

2. **Install LLVM/Clang compiler toolchain** for ARM64

3. **Install OpenBLAS for Windows ARM64**:
   - Via vcpkg (recommended): `vcpkg install openblas:arm64-windows`
   - Or download from: https://github.com/OpenMathLib/OpenBLAS/releases
   - Set `OPENBLAS_ROOT` environment variable to installation path

4. **Install SuiteSparse for Windows ARM64**:
   - Via vcpkg (recommended): `vcpkg install suitesparse:arm64-windows`
   - Or download and build from: https://github.com/DrTimothyAldenDavis/SuiteSparse
   - Set `SUITESPARSE_ROOT` environment variable if needed
   - Required components: LDL, AMD, SuiteSparse_config

**Build Steps:**
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/AlabamaASRL/asset_asrl.git
cd asset_asrl

# Configure with CMake (will detect ARM64 and use OpenBLAS + SuiteSparse)
# Important: If you have multiple Python installations, specify ARM64 Python:
cmake -B build -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR="C:\Python311-ARM64"

# Build
cmake --build build --config Release
```

**Note**: 
- SuiteSparse LDL provides matrix factorization similar to Intel PARDISO
- LDL computes A = L*D*L^T for symmetric indefinite matrices (KKT systems)
- Provides proper inertia information critical for interior-point optimization
- Much more robust than Eigen's sparse solvers for optimization problems

## Documentation
-----

Documentation available at  https://alabamaasrl.github.io/asset_asrl/

## Citation
-----

If you use ASSET in a published work, please cite the following paper: 
```
@inproceedings{Pezent2022,
        author = {James B. Pezent and Jared Sikes and William Ledbetter and Rohan Sood and Kathleen C. Howell and Jeffrey R. Stuart},
        title = {ASSET: Astrodynamics Software and Science Enabling Toolkit},
        booktitle = {AIAA SCITECH 2022 Forum},
        pages = {AIAA 2022-1131},
        year={2022},
        doi = {10.2514/6.2022-1131}
}
```




