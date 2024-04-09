########### Build Cmake ########################
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v3.29/cmake-3.29.0.tar.gz
tar -xzvf cmake-3.29.0.tar.gz
cd cmake-3.29.0/
ls
/bin/bash ./bootstrap --parallel=$1 -- -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_USE_OPENSSL=OFF
make -j$1
make install
## installs to /usr/local/bin/cmake
#########################################