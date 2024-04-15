

# Docker Files
This subfolder contains docker files and supporting scripts for building asset_asrl .whl files for pip installation on linux systems.
Two images are provided based on Ubuntu 18.04 and 20.04. Build wheels on 18.04 for better compatibility with older linux distros.


## Usage
-----
To use, cd to the desired image's folder and build the image. You may optionally set the number of cores used building the image with the
CMakeBuildJobs build argument.
```
cd Ubuntu18.04

docker build -t assetbuild . --build-arg CMakeBuildJobs=12
```
After building the image, start up a new container.

```
docker run -it assetbuild /bin/bash
```

Once inside the container, run the BuildWheel.sh script to build a .whl file for a particular python version (-p), repo branch (-b).
You may also set the number of jobs used when compiling with the -j argument. 

```
bash ~/BuildWheel.sh -b master -j 12 -p 3.10
```
After the build has completed, the packaged wheel file will be available in ~/asset_asrl/wheelhouse. You can then copy this file out of the container
for installation on other systems.







