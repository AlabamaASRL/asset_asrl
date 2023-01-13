# ===============================================================================
#          ___    _____   _____    ______  ______
#         /   |  / ___/  / ___/   / ____/ /_  __/
#        / /| |  \__ \   \__ \   / __/     / /
#       / ___ | ___/ /  ___/ /  / /___    / /
#      /_/  |_|/____/  /____/  /_____/   /_/
#
#  Astrodynamics Software and Science Enabling Toolkit
#
#
#  Dockerfile
#  ----------
#    This builds a minimalist production image of ASSET that includes only
#  runtime dependencies. This image can be used as a base image when creating
#  projects that need to run ASSET in the backend.
#
#
#  Building the Image
#  ------------------
#    This image assumes that ASSET has already been built for x86_64 architecture
#  for python 3.9 and that the asset_asrl/ directory is also available. This 
#  is only a temporary dependency. Once asset_asrl is in PIP/Conda, then we will
#  be able to pull down the ASSET library directly from PIP.
#
#  Firstly, put the ASSET shared object and asset_asrl/ objects in a new "asset_package"
#  directory in the root of this repo:
#
#  asset_asrl/
#    |- asset_package/
#      |- asset.cpython-39-x86-64-linux-gnu.so
#      |- asset_asrl/
#
#  Then, build the image:
#
#    > docker build -t asset -f Dockerfile .
#
# ===============================================================================


FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------
#       Base image updates
# -------------------------------
RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y wget 

# -------------------------------
#            Miniconda
# -------------------------------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh \
  && bash Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -b -p /opt/miniconda3 \
  && /opt/miniconda3/bin/conda init bash \
  && . /root/.bashrc \
  && conda create --name asset \
  && rm Miniconda3-py39_22.11.1-1-Linux-x86_64.sh \
  && echo "conda activate asset" >> /root/.bashrc \
  && conda clean --all

# -------------------------------
#       ASSET Environment
# -------------------------------
RUN . /root/.bashrc \
  && conda config --add channels conda-forge \
  && conda install \
    python=3.9 \
    mkl \
    numpy \
    scipy \
    spiceypy \
  && conda clean --all \
  && echo "ln -s /opt/miniconda3/envs/asset/lib/libomp.so /opt/miniconda3/envs/asset/lib/libomp.so.5" >> /root/.bashrc \
  && echo "export LD_LIBRARY_PATH=/opt/miniconda3/envs/asset/lib:$LD_LIBRARY_PATH" >> /root/.bashrc

# -------------------------------
#             ASSET
# -------------------------------
ADD ./asset_package /root/.local/lib/python3.9/site-packages
