This file contains the instructions for the developers of the library.

To build the documentation using sphinx-doc, it requires the pip packages : 
  pip3 install Cython Sphinx six sphinx-fortran sphinx-rtd-theme sphinxcontrib-matlabdomain

To publish a new release, you have to :
- change the version number in the following files :
  src/calceph.h
  CMakeLists.txt

- update the version of the library using the instructions of
  http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
  in the file :
  CMakeLists.txt

- change the date CALCEPH_RELEASE_DATE_FULL in the file :
  CMakeLists.txt

- update the changelog in the following files :
  NEWS
  doc/source/calceph.final.rst

- apply these previous changes:
  run the commands :
     mkdir build && cd build
     cmake -DBUILD_SHARED_LIBS=ON -DENABLE_DOC=ON -DENABLE_PDF=ON -DENABLE_MEX_OCTAVE=ON -DENABLE_PYTHON=ON  ..
     make && make doc
     CTEST_OUTPUT_ON_FAILURE=1 make test
    
- under mac os, to build the final tarball : perform the following command
  # turn off special handling of ._* files in tar, etc.
  COPYFILE_DISABLE=1   make package_source
  COPYFILE_DISABLE=1   make package_octave 


- build the archive for the Pypi (pip)
  run the commands (create the archive in the directory dist):
  make package_pypi
    
  test the package using testpypi:
    see instructions on https://wiki.python.org/moin/TestPyPI for .pypirc    
    twine upload dist/* -r testpypi
    pip install --user --extra-index-url https://test.pypi.org/pypi calcephpy
    
   release of the package on pypi :
    see instructions on https://wiki.python.org/moin/TestPyPI for .pypirc    
    twine upload dist/*
    pip install --user  calcephpy
   
   
- publish the new version to homebrew
  run the command
     export HOMEBREW_GITHUB_API_TOKEN=....yourtoken....
     cd /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core
     git -C $(brew --repo homebrew/core) checkout main
     brew bump-formula-pr  --sha256 ...shasum_-a_256....tar.gz...  --url https://www.imcce.fr/content/medias/recherche/equipes/asd/calceph/calceph-x.x.x.tar.gz calceph
    

- publish the new version to vcpkg
  see https://learn.microsoft.com/en-us/vcpkg/contributing/maintainer-guide
  replace X.X.X and X_X_X by the major/minor/patch release  
  
  clone the vcpkg project and create a new branch
     git clone git@github.com:....your_github_login.../vcpkg.git
     git checkout -b port_calceph_X_X_X

  edit the following files (update the version and SHA 512)
     ports/calceph/vcpkg.json ports/calceph/portfile.cmake 

  commit the changes
     export MSG_COMMIT='[calceph] update to version X.X.X'
     git add ports/calceph/portfile.cmake  ports/calceph/vcpkg.json
     git commit -m "$MSG_COMMIT"
     vcpkg x-add-version calceph
     git add versions/baseline.json versions/c-/calceph.json
     git commit -m "$MSG_COMMIT"
     git reset --soft HEAD~2 && git commit -m "$MSG_COMMIT"
     git push --set-upstream origin port_calceph_X_X_X

  create the pull request on github

- publish the new version to conda-forge
  clone a fork of https://github.com/conda-forge/calceph-feedstock.git project and create a new branch
  replace X.X.X and X_X_X by the major/minor/patch release  

  clone the vcpkg project and create a new branch
     git clone git@github.com:....your_github_login.../calceph-feedstock.git
     git checkout -b port_calceph_X_X_X

  edit the following files (update the version and SHA 256)
     recipe/meta.yaml

  rerender the feedstock  
     conda install  -c conda-forge conda-build conda-verify conda-smithy mamba conda-index
     conda smithy rerender

  commit the changes
  create the pull request on github


- to detect memory leaks, use the option **-DENABLE_SANITIZER=ON** 