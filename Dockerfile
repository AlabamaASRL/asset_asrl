# ===============================================================================
#          ___    _____   _____    ______  ______
#         /   |  / ___/  / ___/   / ____/ /_  __/
#        / /| |  \__ \   \__ \   / __/     / /
#       / ___ | ___/ /  ___/ /  / /___    / /
#      /_/  |_|/____/  /____/  /_____/   /_/
# 
#  Astrodynamics Software and Science Enabling Toolkit
# ===============================================================================
# 
# Development funded by NASA under Grant No. 80NSSC19K1643
# 
#  Senior Personnel:
#   Rohan Sood            rsood@eng.ua.edu                PI
#   Kathleen Howell       howell@purdue.edu               Co-I
#   Jeff Stuart           jeffrey.r.stuart@jpl.nasa.gov   Co-I
#  Student Contributors:
#   James B. Pezent       jbpezent@crimson.ua.edu         Lead Developer
#   Jared D. Sikes        jdsikes1@crimson.ua.edu         Developer
#   William G. Ledbetter  wgledbetter@crimson.ua.edu      Developer
#   Carrie G. Sandel      cgsandel@crimson.ua.edu         Developer
# ===============================================================================
# 
#  Software Version     : 0.0.1
#  Python   Version     : 3.8
#  System Core Count    : 8
#  System Thread Count  : 8
#  Vectorization Mode   : AVX2 - 256 bit - 4 doubles
#  Linear Solver        : Intel MKL Pardiso
#  Compiled With        : Clang 15.0.6
#  Compiled On/For      : Linux 5.15.0-1026-aws
# ===============================================================================


FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------
#       Base image updates
# -------------------------------
RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y \
    build-essential \
    vim \
    wget \
    cmake \
    git \
    clang \
    lldb \
    libssl-dev \
    llvm \
    lld \
    software-properties-common

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
#              MKL
# -------------------------------
RUN wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19138/l_onemkl_p_2023.0.0.25398_offline.sh \
  && sh ./l_onemkl_p_2023.0.0.25398_offline.sh -a --silent --eula accept \
  && rm l_onemkl_p_2023.0.0.25398_offline.sh \
  && echo ". /opt/intel/oneapi/setvars.sh" >> /root/.bashrc

# -------------------------------
#       ASSET Environment
# -------------------------------
RUN . /root/.bashrc \
  && conda config --add channels conda-forge \
  && conda install \
    python=3.8 \
    nomkl \
    numpy \
    scipy \
    pandas \
    plotly \
    matplotlib \
    ipython \
    spiceypy

# -------------------------------
#        LLVM & CLANG 15
# -------------------------------
RUN wget https://apt.llvm.org/llvm.sh \
  && chmod +x llvm.sh \
  && ./llvm.sh 15 all

# -------------------------------
#             ASSET
# -------------------------------
RUN . /root/.bashrc \ 
  && export CC=/usr/bin/clang-15 \
  && export CXX=/usr/bin/clang++-15 \
  && git clone https://github.com/AlabamaASRL/asset_asrl.git \
  && cd asset_asrl \
  && git submodule update --init --recursive \
  && mkdir build \
  && cd build/ \
  && cmake -DCMAKE_BUILD_TYPE=Release .. \
  && cmake --build . -j6

