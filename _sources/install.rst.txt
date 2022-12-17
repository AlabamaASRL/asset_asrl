Installation
============

ASSET is packaged using the standard python conventions, which makes getting started

Dependencies
------------

Most of ASSET's dependencies are automatically installed during the ``pip install`` process, or are included as submodules in the git repository.
However, some libraries do require manual installation.
Intel's Math Kernel Library (MKL) is mandatory for users on x86 systems, and although only officially supported on Intel, our dev team has seen comparable performance using AMD.
If you are using an ARM architecture (Apple M1)

MKL
^^^

In our opinion, the easiest way to install MKL is using `Intel's Anaconda channel <https://anaconda.org/intel>`_.
For the default configuration, run ``conda install -c intel mkl-devel mkl-static`` to download the static MKL libraries for your platform.
Then you must set the ``MKLROOT`` environment variable to point to the install location.
On Unix-based systems, this will be the root of your current conda environment, which can be identified by running ``which python``.
For example, your ``MKLROOT`` may be ``~/.conda/envs/asset/``.

Doxygen
^^^^^^^

Again, doxygen is only required if you plan to build documentation locally.
