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




