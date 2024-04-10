
ncores=10
branch=master
PyVer=3.10

while getopts j:b:p: flag
do
    case "${flag}" in
        j) ncores=${OPTARG};;
        b) branch=${OPTARG};;
        p) PyVer=${OPTARG};;
    esac
done

##############################################
cd ~/

if [ -d "~/asset_asrl" ]; then
    echo "asset_asrl already exists"
else
    git clone https://github.com/AlabamaASRL/asset_asrl
    cd asset_asrl
    git submodule update --init --recursive
fi

cd ~/asset_asrl

git checkout $branch

############################################
BuildEnv=__build$PyVer

if conda info --envs | grep -q $BuildEnv; then echo "$BuildEnv already exists";else conda create -y -n $BuildEnv python=$PyVer; fi

conda init bash
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate $BuildEnv

pip install patchelf auditwheel
#############################################

rm -rf build

/usr/local/bin/cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DPIP_INSTALL=True -DBUILD_ASSET_WHEEL=True -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
/usr/local/bin/cmake --build build --target asset pypiwheel --config Release -- -j8 

auditwheel show build/pypiwheel/asset_asrl/dist/*.whl  ## manylinux_2_27_x86_64 
auditwheel repair --plat manylinux_2_27_x86_64 build/pypiwheel/asset_asrl/dist/*.whl

conda activate base

cd ~/