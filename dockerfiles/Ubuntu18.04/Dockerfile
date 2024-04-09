FROM ubuntu:18.04

ARG CmakeBuildJobs=8


RUN apt-get update -y
RUN apt-get install git wget build-essential clang lld llvm dos2unix nano -y



COPY ./CMakeInstall.sh /tmp/CMakeInstall.sh
COPY ./CondaInstall.sh /tmp/CondaInstall.sh
COPY ./MKLInstall.sh   /tmp/MKLInstall.sh
COPY ./BuildWheel.sh   /tmp/BuildWheel.sh
COPY ./BuildAllWheels.sh   /tmp/BuildAllWheels.sh



RUN dos2unix /tmp/CMakeInstall.sh
RUN dos2unix /tmp/MKLInstall.sh
RUN dos2unix /tmp/CondaInstall.sh
RUN dos2unix /tmp/BuildWheel.sh
RUN dos2unix /tmp/BuildAllWheels.sh

RUN /bin/bash /tmp/CMakeInstall.sh $CmakeBuildJobs
RUN /bin/bash /tmp/MKLInstall.sh
RUN /bin/bash /tmp/CondaInstall.sh

