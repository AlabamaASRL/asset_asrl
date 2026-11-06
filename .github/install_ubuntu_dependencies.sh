set -e

# Update
sudo apt update
# sudo apt upgrade

# Install Compiler
sudo apt install $COMPILER
$COMPILER -v
echo "CC=$COMPILER" >>$GITHUB_ENV
COMPILER_NAME=${COMPILER%-*}
COMPILER_VERSION=${COMPILER#*-}
if [[ $COMPILER_NAME == "clang" ]]; then
  echo "CXX=$COMPILER_NAME++-$COMPILER_VERSION" >>$GITHUB_ENV
else
  echo "CXX=g++-$COMPILER_VERSION" >>$GITHUB_ENV
fi

# Install MKL
cd /tmp
wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo install -D -m 0644 GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB.gpg /etc/apt/keyrings/intel-oneapi.gpg
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB.gpg
echo "deb [signed-by=/etc/apt/keyrings/intel-oneapi.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel intel-oneapi-mkl intel-oneapi-openmp

#source /opt/intel/oneapi/mkl/latest/env/vars.sh
source /opt/intel/oneapi/setvars.sh
#source /opt/intel/oneapi/latest/oneapi-vars.sh


echo "MKLROOT=$MKLROOT" >>$GITHUB_ENV
echo "INTEL=/opt/intel/oneapi" >>$GITHUB_ENV
echo "ONEAPI_ROOT=$ONEAPI_ROOT" >>$GITHUB_ENV

# Install other dependencies
sudo apt install cmake
