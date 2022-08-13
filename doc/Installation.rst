Python: Installing ASSET
========================


Windows Binary Dependencies
---------------------------
For windows binary distributions installable via pip or downloaded from github, we statically link intel's math kernel library into python pyd file, thus
it is not sctrictly neccessary to have mkl installed on the target system. However, it is still neccessary to download and install the LLVM toolchain, and
place it's redistributable LLVM/bin folder on the path. This ensures that the asset python library has access to two neccessary DLLs that are not statically
linked (vcruntime140.dll and libiomp5md.dll).


Installing ASSET is the first step in incorporating it into your astrodynamics work flow.
ASSET for Windows requires that the `LLVM toolkit <https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/LLVM-11.0.1-win64.exe>`_ (this downloads the LLVMx64 installer for Windows).
When installing LLVM be sure to add it to the system's path.
If the above step is not completed correctly, Python will be unable to load the library.

Next, to access ASSET from Python, take the ASSET pyd file provided to you and place it in the same directory as your current Python work flow.

.. warning::
	Unfortunately due to potential conflicts with system path variables, using ASSET inside of a Python virtual environment may result in a failure to load or function correctly.
	If you are having issues loading the library or any associated DLLs, ensure you are not in a Python virtual environment.

	However, Docker containers have been used in the past successfully. Regardless, if you are having issues with installing ASSET, verify first a correct installation outside of any
	virtual environment.

Now ASSET should be available to include into any Python files within the directory where the ASSET .pyd file is located.


.. code-block:: python

	import asset as ast

Now with ASSET available in your Python directory, the next step is to learn the conventions necessary to construct functions for use with ASSET.
Head over to :ref:`Python: Vector Functions`.

.. warning::
	Some issues have been found with conflicting libomp.dll and libiomp5md.dll on **Windows** machines.
	A few potential fixes for this are as follows:

	* Update or reinstall numpy for your current Python installation.
	* If the above does not resolve the issues, adding this line of code to your Python file in the import section.

.. code-block:: python

		import os
		os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

Adding this environment variable to your Python environment may also resolve the issue.
