Installing ASSET
================

This tutorial will guide you through the complete setup process for installing and building ASSET from scratch on:
* Windows
* Linux
* Docker

Windows Installation
--------------------

In order to build ASSET on Windows the following dependencies are required:

* Visual Studio 2017 or greater `(Link to Visual Studio Community 2022) <https://visualstudio.microsoft.com/downloads/>`_ 
* `LLVM Compiler Toolkit <https://github.com/llvm/llvm-project/releases/latest>`_
* `Intel oneAPI MKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=window&distributions=offline>`_
* A Python installation with Numpy, however we recommend using Anaconda.
	* `Anaconda <https://www.anaconda.com/products/distribution>`_
* The Python packages SpiceyPy, Sphinx, and Furo


Step-By-Step Guide
^^^^^^^^^^^^^^^^^^
#. We will begin by installing Visual Studio (VS). 

	  ..  note:: 
  
	  Follow the VS installation prompts until you are prompted to select any additional packages to install with VS. 
	  Here we will want to select **Desktop development with C++**; however, we are going to make one change to the defaults for the C++ install and **un-check** the option for "C++ Clang tools for Windows" (shown below).

	  .. image:: _static/vsstudiodesktopc++.png

	  Also ensure that the "**C++ CMake Tools for Windows**" is checked.
	  After doing this continue and finish the installation.

		.. image:: _static/VSoptions.PNG

#. Next we will install the LLVM Compiler Toolkit. An LLVM installation of at least LLVM-11 is required, and we recommend using the latest version if possible.

	  ..  note:: 
  
	  Navigate to the LLVM github releases page, select your desired version, and download the LLVM win64 installer (LLVM-15.0.0-win64.exe for example).
	  Proceed with the LLVM install, however ensure that the install option "**Do not add LLVM to the system PATH**" is checked.
	  After finishing the LLVM installation we can proceed with installing Intel MKL.

		.. image:: _static/LLVMnopath.PNG

#. Download and install the Intel oneAPI MKL, using the offline installer.

	  ..  note:: 
  
	  Install with the recommended settings and then proceed to installing Anaconda, if desired.
	  If you are not using Anaconda you may skip the next step.

#. Now download and install Anaconda.

	  ..  note:: 
	  
	  Check that Anaconda is set to be placed on the PATH and set Anaconda to be your default Python.  Additionally install the following packages.

	  * SpiceyPy, Sphinx, and Furo

	  Your version of Anaconda may be installed in "ProgramData" and may be named "Anaconda", "Anaconda3", or "anaconda3", depending on your installation settings.

	  .. image:: _static/anacondainstall.PNG

	  
	  After installing anaconda, it may be necessary to upgrade numpy through pip, due to potential compatibibility issues with numpy's version of libiomp5md.dll and Intel MKL's version.

	  .. code-block:: console

		pip install numpy --upgrade

	  .. warning::

	    if you aren't using anaconda, you are still going to need numpy (comes with anaconda by default).


#. Next we will be adding the required system and PATH variables to build ASSET.

	  ..  note:: 
  
	  First, add the Intel oneAPI to a new system variable named "ONEAPI_ROOT" and set the variable value to your oneAPI install directory, as shown below.
	  If your install directory is different, be sure to point this to the correct folder. This may have already been set when you installed oneAPI.

		.. image:: _static/oneapiroot.PNG

	  .. warning::

		If you are using an Alder Lake or Raptor Lake Intel CPU it may be beneficial to add the system variable "MKL_ENABLE_INSTRUCTIONS" with value "AVX" as well. 

#. We also need to add a few variables to our Path. 

	  ..  note:: 
  
	  The following directories should be added to your system PATH and set as your default Python install.
	  Once again, be sure to point this to the actual location of the relevant folders on your computer.

		.. image:: _static/anacondapath.PNG

#. Now, clone the `ASSET repo <https://github.com/AlabamaASRL/asset_asrl>`_  to where you want it to live.

	  .. note::

	  If you are cloning from the command line, make sure to call :code:`git submodule update --init --recursive` inside of the repository after cloning to ensure that the dependencies are downloaded.


#. We are almost there! Now, open VS and be sure to run it as administrator to avoid any conflicts when building the Python library. Additionally, ensure that your Python IDE is closed during this step. 
	 
	  .. note::

	  Open the ASSET folder with VS and wait for it to finish loading. After VS has finished loading the ASSET repo directory, navigate to the CMakeSettings.json file. Here it is recommended to change the integer number of threads to be the amount of RAM your computer has divided by 3 (rounded down).

	  .. image:: _static/cmakejson.PNG

	  .. image:: _static/threads.PNG

	  Now we are going to configure the ASSET CMake settings by going to the project dropdown menu and selecting "Configure ASSET" or "Configure Cache" (depending on the version of VS).
	  Wait until the output message from CMake says that it is finished and proceed to the build step.

	  .. image:: _static/config.png

	  The last step is to actually build ASSET! After the configuration step has completed navigate to the Build dropdown menu and choose "Build All" (or Ctrl + Shift + B). Keep your Python IDE closed until the build completes.

	  .. image:: _static/build.png

#. With that you should have a successfully built ASSET Python library and are ready to get started with the rest of the tutorials.

	  .. note::

	  To import ASSET simply use the following in your Python IDE

	  .. code-block:: python

		import asset_asrl



Linux Installation
------------------
The dependencies for Linux installations are similar to that of Windows; however, we'll be using Visual Studio Code for our IDE:

* `Visual Studio Code <https://code.visualstudio.com/download>`_
	* `C/C++ Extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools>`_ 
	* `CMake Tools Extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools>`_
	* Ensure that you have a working version of `CMake installed <https://cmake.org/install/>`_
* `LLVM Compiler Toolkit (recommended) <https://github.com/llvm/llvm-project/releases/latest>`_
	* GCC 9 or greater can also be used
* `Intel oneAPI MKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&distributions=offline>`_
* `Anaconda <https://www.anaconda.com/products/distribution#linux>`_
* The Python packages SpiceyPy, Sphinx, and Furo

Step-by-Step Guide
^^^^^^^^^^^^^^^^^^
This guide was written assuming that the user has a working Ubuntu installation; however, ASSET will work with other Linux distributions. Simply use the appropriate commands to install the required packages.
If it is desired to use an IDE other than Visual Studio Code, it is still required that a version of CMake of at least 3.16 is installed.

#. We will begin by installing Visual Studio Code (VSCode). 

	  ..  note:: 
  
	  Install VSC as desired, along with the **C/C++** and **CMake Tools** Extensions.

#. Now, install LLVM clang using the package manager.

	.. code-block:: console

		sudo apt install clang lldb lld
	
	* or if you desire to use gcc
	

	  .. code-block:: console

		sudo apt update
		sudo apt install build-essential
		gcc --version

#. Download and install the Intel oneAPI MKL, using the offline installer. The complete Linux installation guide for Intel oneAPI can be found `here <https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top.html>`_.

	  ..  note:: 

		We recommend that you use the oneAPI gui installer through the Intel website; however, if it necessary to install via the bash terminal, those directions can be found `here <https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html#apt>`_.

		Install with the recommended settings and then proceed to installing Anaconda, if desired.
		If you are not using Anaconda you may skip the next step.

#. Now, download and install Anaconda.

      * Use pip to install SpiceyPy.

	  .. warning::

	  if you are using anaconda rather than a generic python installation, it may be necessary to upgrade numpy through pip, due to potential compatibibility issues with numpy's version of openmp and Intel MKL's version.

	  .. code-block:: console

		conda install nomkl

	  Even if you already have anaconda or a python distribution installed you will need these packages.

	  * SpiceyPy, Sphinx, and Furo

	  .. warning::

	    if you aren't using anaconda, you are still going to need numpy (comes with anaconda by default).

#. Next, we will set the required system variables to build ASSET.

	  ..  note:: 
  
	  Use the source command to properly setup the oneAPI variables.

	  .. code-block:: console

	   source /opt/intel/oneapi/setvars.sh

#. Now, clone the `ASSET repo <https://github.com/AlabamaASRL/asset_asrl>`_  to where you want it to live.

	.. note::

		  If you are cloning from the command line, make sure to call :code:`git submodule update --init --recursive` inside of the repository after cloning to ensure that the dependencies are downloaded.


#. After installing the dependencies, open VSCode to build ASSET.
	
	  .. note::

	  Configure the VSCode CMake extension to use the Ninja generator. You may need to install Ninja.

	  .. code-block:: console

		apt install ninja-build

	  Select the type of build (1) you wish to perform (Release is correct if you are going to be running code using ASSET), and kit to use (2) (gcc or clang).

	  .. image:: _static/vscodevariant.PNG

	  Now hit build to begin building ASSET

	  .. image:: _static/vscodebuild.png

#. With that you should have a successfully built ASSET Python library and are ready to get started with the rest of the tutorials.

	  .. note::

	  To import ASSET simply use the following in your Python IDE

	  .. code-block:: python

		import asset_asrl

Docker
------

Want to skip installing all of the dependencies? ASSET can be installed through a Docker image. There's also great integrations of Docker and VS Code.

Download the Docker/VS Code project template here and get to coding!

https://github.com/jasonmeverett/asset-example-project

Developing Docker Images 
^^^^^^^^^^^^^^^^^^^^^^^^

Interested in contributing to Docker/ASSET development? Check in-depth details in `Dockerfile` and `Dockerfile-dev` in the repository.



Next Steps
----------
If this is your first time using ASSET, it's now recommended that you begin reviewing the remaining tutorials, such as :ref:`Vector Function Tutorial`. After you feel comfortable with some of ASSET's coding paradigms, a select set
of tutorials that highlight ASSET's features and capabilities are provided in :ref:`Examples`.





		




