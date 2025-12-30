
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
- **Windows ARM64**: Limited support - builds from source without Intel MKL

### Building from Source on Windows ARM64

On Windows ARM64 systems, Intel's oneAPI Math Kernel Library (MKL) is not available. When building from source on these systems, ASSET will automatically use Eigen's built-in sparse solvers as an alternative. While this configuration works, performance may be reduced compared to MKL-enabled builds.

To build from source on Windows ARM64:
1. Install LLVM/Clang compiler toolchain
2. Clone the repository with submodules: `git clone --recursive https://github.com/AlabamaASRL/asset_asrl.git`
3. Configure and build with CMake (MKL will be automatically skipped)

**Note**: The optimizer (PSIOPT) will use Eigen's SimplicialLDLT solver instead of Intel PARDISO when MKL is not available.

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




