FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------
#       Base image updates
# -------------------------------
RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y \
    wget \
    cmake \
    git \
    software-properties-common

# -------------------------------
#        LLVM & CLANG 15
# -------------------------------
RUN wget https://apt.llvm.org/llvm.sh \
  && chmod +x llvm.sh \
  && ./llvm.sh 15 all \
  && rm -rf ./llvm.sh

# -------------------------------
#            Miniconda
# -------------------------------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh \
  && bash Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -b -p /opt/miniconda3 \
  && /opt/miniconda3/bin/conda init bash \
  && /opt/miniconda3/bin/conda create --name asset \
  && rm Miniconda3-py39_22.11.1-1-Linux-x86_64.sh \
  && echo "conda activate asset" >> /root/.bashrc

# -------------------------------
#       ASSET Environment
# -------------------------------
RUN . /root/.bashrc \
  && conda config --add channels conda-forge \
  && conda install \
    python=3.9 \
    mkl \
    mkl-include \
    numpy \
    scipy \
    spiceypy

# -------------------------------
#             ASSET
# -------------------------------
RUN . /root/.bashrc \ 
  && export CC=/usr/bin/clang-15 \
  && export CXX=/usr/bin/clang++-15 \
  && export MKLROOT=/opt/miniconda3/envs/asset \
  && git clone https://github.com/AlabamaASRL/asset_asrl.git \
  && cd asset_asrl \
  && git submodule update --init --recursive \
  && mkdir build \
  && cd build/ \
  && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=true .. \
  && cmake --build . -j4 \
  && cd / \
  && rm -rf /asset_asrl

