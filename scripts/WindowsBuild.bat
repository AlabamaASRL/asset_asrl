

$v1  = '-DCMAKE_BUILD_TYPE=Release' 
$v2  = '-DPIP_INSTALL=True '
$v3  = '-DBUILD_SPHINX_DOCS=True' 
$v4  = '-DCMAKE_C_COMPILER=C:/Program Files/LLVM/bin/clang-cl.exe' 
$v5  = '-DCMAKE_C_COMPILER=C:/Program Files/LLVM/bin/clang-cl.exe' 
$v6  = '-DCMAKE_CXX_COMPILER=C:/Program Files/LLVM/bin/clang-cl.exe' 
$v7  = '-DCMAKE_LINKER=C:/Program Files/LLVM/bin/lld-link.exe' 
$v8  = '-DCMAKE_CXX_COMPILER_RANLIB=C:/Program Files/LLVM/bin/llvm-ranlib.exe' 
$v9  = '-DCMAKE_CXX_COMPILER_AR=C:/Program Files/LLVM/bin/llvm-ar.exe' 
$v10 = '-DCMAKE_C_COMPILER_RANLIB=C:/Program Files/LLVM/bin/llvm-ranlib.exe' 
$v11 = '-DCMAKE_C_COMPILER_AR=C:/Program Files/LLVM/bin/llvm-ar.exe' 



cmake -S . -B build -G "Ninja" $v1 $v2 $v3 $v4 $v5 $v6 $v7 $v8 $v9 $v10 $v11

cmake --build build --target asset pypiwheel --config Release -- -j24