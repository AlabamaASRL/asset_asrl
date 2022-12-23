.. ASSET documentation master file, created by
   sphinx-quickstart on Mon Jan 27 14:56:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ASSET's documentation!
=================================

ASSET (Astrodynamics Software and Science Enabling Toolkit) is a modular, extensible library for trajectory design and optimal control.
It uses a custom implementation of vector math formalisms to enable rapid implementation of dynamical systems and automatic differentiation.
The phase object is the core of the optimal control functionality, and by linking multiple phases together, the user can construct scenarios of arbitrary complexity.
A newly developed high-performance interior-point optimizer (PSIOPT) is included with the library, which enables quick turnaround from concept to solution.


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    tutorials/tutorials.rst
    examples/examples.rst
    python/python.rst

Citation
========

Cite the following `paper <https://www.researchgate.net/publication/357567956_ASSET_Astrodynamics_Software_and_Science_Enabling_Toolkit>`_ if you used ASSET in your work.

.. note::
    .. raw:: latex

    @inproceedings{Pezent2022,
        author = {James B. Pezent and Jared Sikes and William Ledbetter and Rohan Sood and Kathleen C. Howell and Jeffrey R. Stuart},
        title = {ASSET: Astrodynamics Software and Science Enabling Toolkit},
        booktitle = {AIAA SCITECH 2022 Forum},
        pages = {AIAA 2022-1131},
        year={2022},
        doi = {10.2514/6.2022-1131}
    }



If you have questions, please email any of:

    jbpezent@crimson.ua.edu

    jdsikes1@crimson.ua.edu

    wgledbetter@crimson.ua.edu


..
    cpp/cpp_root.rst
    install
    contrib



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
