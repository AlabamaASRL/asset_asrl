FROM ubuntu:20.04

ARG CMakeBuildJobs=8


RUN apt-get update -y
RUN apt-get install git wget build-essential clang-12 lld-12 llvm-12 dos2unix nano -y



COPY ./CMakeInstall.sh /tmp/CMakeInstall.sh
COPY ./CondaInstall.sh /tmp/CondaInstall.sh
COPY ./MKLInstall.sh   /tmp/MKLInstall.sh
COPY ./BuildWheel.sh   /root/BuildWheel.sh
COPY ./BuildAllWheels.sh   /root/BuildAllWheels.sh



RUN dos2unix /tmp/CMakeInstall.sh
RUN dos2unix /tmp/MKLInstall.sh
RUN dos2unix /tmp/CondaInstall.sh
RUN dos2unix /root/BuildWheel.sh
RUN dos2unix /root/BuildAllWheels.sh

RUN /bin/bash /tmp/CMakeInstall.sh $CMakeBuildJobs
RUN /bin/bash /tmp/MKLInstall.sh
RUN /bin/bash /tmp/CondaInstall.sh

