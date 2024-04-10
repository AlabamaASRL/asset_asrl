PythonVersions=("3.7" "3.8" "3.9" "3.10" "3.11")
ncores=10
branch=master


while getopts j:b: flag
do
    case "${flag}" in
        j) ncores=${OPTARG};;
        b) branch=${OPTARG};;
    esac
done


for pyver in ${PythonVersions[@]}
 do
        echo "Building Python $pyver wheel"
        bash ~/BuildWheel.sh -j $ncores -b $branch -p $pyver
        echo "Finished building Python $pyver wheel"

 done
 
