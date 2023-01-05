# ASSET: Astrodynamics Software and Science Enabling Toolkit

ASSET (Astrodynamics Software and Science Enabling Toolkit) is a modular, extensible library for trajectory design and optimal control.
It uses a custom implementation of vector math formalisms to enable rapid implementation of dynamical systems and automatic differentiation.
The phase object is the core of the optimal control functionality, and by linking multiple phases together, the user can construct scenarios of arbitrary complexity.
A newly developed high performance interior-point optimizer (PSIOPT) is included with the library, which enables quick turnaround from concept to solution.

Development funded by NASA under Grant No. 80NSSC19K1643.

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

## [Docker](https://docs.docker.com/get-docker/)
-----

ASSET can be ran in a Docker container. When using the docker container, there's no need to install dependencies manually. Just pull (or build) the most recent Docker image and use it.

### **Pull down Development Image**
----

A working version of the Docker image can be pulled from [Docker Hub]() to skip having to manually build the docker image. To get this image:

```
docker pull jasonmeverett/asset_asrl:1.0
```

Then, to start up the container and run ASSET:

```
docker run -it jasonmeverett/asset_asrl:1.0 bash
cd /asset_asrl/
python AnalyticExample.py
```

### **Build Docker Image from Scratch**
----

The ASSET Docker image can also be built manually.

* Pull down the most recent main repository.
* Open a terminal in this folder and run:

```
docker build -t asset:1.0 .
```

This will build the Docker image containing ASSET and all of its dependencies. ASSET examples will be available in the image at `/asset_asrl` in the Docker container.


### **Using Docker Image with new Project Files**
----

If you'd like to use ASSET for a new project (in a different repository), you can mount your project files into the Docker container, which will allow you to utilize ASSET in the Docker container while developing files on your local machine.

Say you have a directory where you are working on some new Python scripts that utilize ASSET:

```
~/
  myProject/
    assetScript1.py
    assetScript2.py
    ...
```

If you want the `myProject` folder available from within the Docker image, you can "mount" the folder into the Docker image as follows:

```
cd myProject/
docker run -it -v "$(pwd):/myProject" asset:1.0 bash
```

Now, all files in `myProject/` will be available at `/myProject/` in the Docker container. These two folders will be "mirrored" (any change made in Docker/local will be instantly replecated in the other environment).

### **Docker and VS Code**
----

VS Code as several useful Docker extensions that make developing with Docker seemless.

*TODO: Fill out this section*


