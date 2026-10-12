
@echo off
setlocal EnableExtensions
title ASSET Windows Build

REM ============================================================
REM 0. Always run from the directory containing this script
REM ============================================================

cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Cannot enter the script directory.
    goto :fail
)

echo.
echo ============================================================
echo                  ASSET Windows Build
echo ============================================================
echo.
echo Repository: %CD%
echo.

if not exist "CMakeLists.txt" (
    echo ERROR: CMakeLists.txt was not found in %CD%
    echo Place this script in the ASSET repository root.
    goto :fail
)

REM ============================================================
REM 1. Check Visual Studio C++ Build Tools
REM ============================================================

echo.
echo [1/8] Checking Visual Studio C++ Build Tools...

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_PATH="

if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%V in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_PATH=%%V"
)

if defined VS_PATH goto :vs_found

echo Visual Studio C++ tools are missing.
call :check_winget
if errorlevel 1 goto :fail

echo Installing Visual Studio 2022 Build Tools...
winget install --exact --id Microsoft.VisualStudio.2022.BuildTools --source winget --accept-source-agreements --accept-package-agreements --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

if errorlevel 1 (
    echo ERROR: Visual Studio Build Tools installation failed.
    goto :fail
)

if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%V in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_PATH=%%V"
)

if not defined VS_PATH (
    echo ERROR: C++ Build Tools were not detected after installation.
    echo Complete any pending installation and rerun this script.
    goto :fail
)

:vs_found
echo Found Visual Studio: %VS_PATH%

if not exist "%VS_PATH%\Common7\Tools\VsDevCmd.bat" (
    echo ERROR: VsDevCmd.bat was not found.
    goto :fail
)

call "%VS_PATH%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 (
    echo ERROR: Failed to initialize Visual Studio.
    goto :fail
)

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: MSVC compiler is unavailable.
    goto :fail
)


REM ============================================================
REM 3. Check LLVM
REM ============================================================

echo.
echo [3/8] Checking LLVM compiler and linker...

call :find_llvm
if defined LLVM_BIN goto :llvm_found

echo LLVM was not found.
call :check_winget
if errorlevel 1 goto :fail

echo Attempting to install LLVM 18.1.8...
winget install --exact --id LLVM.LLVM --version 18.1.8 --source winget --accept-source-agreements --accept-package-agreements

REM Check again regardless of installer exit code.
call :find_llvm
if defined LLVM_BIN goto :llvm_found

echo.
echo ERROR: clang-cl.exe and lld-link.exe were not found.
echo Install a compatible LLVM version and rerun this script.
echo Expected default path: C:\Program Files\LLVM\bin
goto :fail

:llvm_found
echo LLVM directory: %LLVM_BIN%

REM ============================================================
REM 4. Check and install CMake and Ninja
REM ============================================================

echo.
echo [4/8] Checking CMake and Ninja...

set "PYTHON_SCRIPTS="
for /f "delims=" %%V in ('%PYTHON% -c "import sysconfig; print(sysconfig.get_path('scripts'))"') do set "PYTHON_SCRIPTS=%%V"

set "PATH=%PYTHON_SCRIPTS%;%PATH%"

set "NEED_CMAKE="
set "NEED_NINJA="

where cmake.exe >nul 2>&1
if errorlevel 1 set "NEED_CMAKE=1"

where ninja.exe >nul 2>&1
if errorlevel 1 set "NEED_NINJA=1"

if not defined NEED_CMAKE if not defined NEED_NINJA goto :build_tools_found

if defined NEED_CMAKE echo CMake is missing.
if defined NEED_NINJA echo Ninja is missing.

if defined NEED_CMAKE (
    %PYTHON% -m pip install cmake
    if errorlevel 1 (
        echo ERROR: Could not install CMake.
        goto :fail
    )
)

if defined NEED_NINJA (
    %PYTHON% -m pip install ninja
    if errorlevel 1 (
        echo ERROR: Could not install Ninja.
        goto :fail
    )
)

set "PATH=%PYTHON_SCRIPTS%;%PATH%"

where cmake.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: CMake is still unavailable.
    goto :fail
)

where ninja.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: Ninja is still unavailable.
    goto :fail
)

:build_tools_found
echo CMake and Ninja are available.

REM ============================================================
REM 5. Check MKL headers and libraries before installing
REM ============================================================

echo.
echo [5/8] Checking Intel MKL...

for /f "delims=" %%V in ('%PYTHON% -c "import sys; print(sys.prefix)"') do set "PYTHON_PREFIX=%%V"

set "MKLROOT=%PYTHON_PREFIX%\Library"
set "MKL_INCLUDE_DIRS=%MKLROOT%\include"
set "MKL_LIB_DIR=%MKLROOT%\lib"

set "MKL_INTERFACE_LIBRARY=%MKL_LIB_DIR%\mkl_intel_lp64_dll.lib"
set "MKL_SEQTHR_LIBRARY=%MKL_LIB_DIR%\mkl_sequential_dll.lib"
set "MKL_CORE_LIBRARY=%MKL_LIB_DIR%\mkl_core_dll.lib"

set "NEED_MKL="

if not exist "%MKL_INCLUDE_DIRS%\mkl.h" set "NEED_MKL=1"
if not exist "%MKL_INTERFACE_LIBRARY%" set "NEED_MKL=1"
if not exist "%MKL_SEQTHR_LIBRARY%" set "NEED_MKL=1"
if not exist "%MKL_CORE_LIBRARY%" set "NEED_MKL=1"

if not defined NEED_MKL goto :mkl_found

echo MKL headers or libraries are missing.
echo Installing mkl-devel...
%PYTHON% -m pip install mkl-devel
if errorlevel 1 (
    echo ERROR: Could not install mkl-devel.
    goto :fail
)

REM Recalculate the Python prefix and paths after installation.
for /f "delims=" %%V in ('%PYTHON% -c "import sys; print(sys.prefix)"') do set "PYTHON_PREFIX=%%V"

set "MKLROOT=%PYTHON_PREFIX%\Library"
set "MKL_INCLUDE_DIRS=%MKLROOT%\include"
set "MKL_LIB_DIR=%MKLROOT%\lib"
set "MKL_INTERFACE_LIBRARY=%MKL_LIB_DIR%\mkl_intel_lp64_dll.lib"
set "MKL_SEQTHR_LIBRARY=%MKL_LIB_DIR%\mkl_sequential_dll.lib"
set "MKL_CORE_LIBRARY=%MKL_LIB_DIR%\mkl_core_dll.lib"

if not exist "%MKL_INCLUDE_DIRS%\mkl.h" (
    echo ERROR: MKL header is still missing.
    goto :fail
)
if not exist "%MKL_INTERFACE_LIBRARY%" (
    echo ERROR: MKL interface library is still missing.
    goto :fail
)
if not exist "%MKL_SEQTHR_LIBRARY%" (
    echo ERROR: MKL sequential library is still missing.
    goto :fail
)
if not exist "%MKL_CORE_LIBRARY%" (
    echo ERROR: MKL core library is still missing.
    goto :fail
)

:mkl_found
echo MKL headers and libraries are available.

REM ============================================================
REM 6. Check Intel OpenMP Python package
REM ============================================================

echo.
echo [6/8] Checking Intel OpenMP...

%PYTHON% -m pip show intel-openmp >nul 2>&1
if not errorlevel 1 goto :openmp_found

echo Intel OpenMP package is missing. Installing...
%PYTHON% -m pip install intel-openmp
if errorlevel 1 (
    echo ERROR: Could not install intel-openmp.
    goto :fail
)

:openmp_found
echo Intel OpenMP package is available.

set "MKL_LIBRARIES=%MKL_INTERFACE_LIBRARY%;%MKL_SEQTHR_LIBRARY%;%MKL_CORE_LIBRARY%"

REM ============================================================
REM 7. Configure CMake
REM ============================================================

echo.
echo [7/8] Configuring CMake...

cmake -S . -B build -G Ninja ^
    "-DCMAKE_BUILD_TYPE=Release" ^
    "-DPIP_INSTALL=True" ^
    "-DBUILD_SPHINX_DOCS=True" ^
    "-DCMAKE_C_COMPILER=%LLVM_BIN%clang-cl.exe" ^
    "-DCMAKE_CXX_COMPILER=%LLVM_BIN%clang-cl.exe" ^
    "-DCMAKE_LINKER=%LLVM_BIN%lld-link.exe" ^
    "-DMKLROOT=%MKLROOT%" ^
    "-DMKL_INCLUDE_DIRS=%MKL_INCLUDE_DIRS%" ^
    "-DMKL_INTERFACE_LIBRARY=%MKL_INTERFACE_LIBRARY%" ^
    "-DMKL_SEQTHR_LIBRARY=%MKL_SEQTHR_LIBRARY%" ^
    "-DMKL_CORE_LIBRARY=%MKL_CORE_LIBRARY%" ^
    "-DMKL_LIBRARIES=%MKL_LIBRARIES%"

if errorlevel 1 (
    echo ERROR: CMake configuration failed.
    goto :fail
)

REM ============================================================
REM 8. Build ASSET
REM ============================================================

echo.
echo [8/8] Building ASSET and Python wheel...

cmake --build build --target asset pypiwheel --parallel 2
if errorlevel 1 (
    echo ERROR: ASSET build failed.
    goto :fail
)

echo.
echo ============================================================
echo BUILD COMPLETED
echo ============================================================
echo.
echo Generated wheel files:
where /r build *.whl
echo.
pause
exit /b 0

REM ============================================================
REM Helper: locate LLVM
REM ============================================================

:find_llvm
set "LLVM_BIN="

for /f "delims=" %%V in ('where clang-cl.exe 2^>nul') do (
    if not defined LLVM_BIN set "LLVM_BIN=%%~dpV"
)

if not defined LLVM_BIN if exist "C:\Program Files\LLVM\bin\clang-cl.exe" (
    if exist "C:\Program Files\LLVM\bin\lld-link.exe" (
        set "LLVM_BIN=C:\Program Files\LLVM\bin\"
    )
)

if defined LLVM_BIN (
    if not exist "%LLVM_BIN%clang-cl.exe" set "LLVM_BIN="
    if defined LLVM_BIN if not exist "%LLVM_BIN%lld-link.exe" set "LLVM_BIN="
)

exit /b 0

REM ============================================================
REM Helper: check winget only when an installation is needed
REM ============================================================

:check_winget
where winget.exe >nul 2>&1
if not errorlevel 1 exit /b 0

echo ERROR: winget is required to install missing prerequisites.
echo Install or update App Installer from the Microsoft Store.
exit /b 1

REM ============================================================
REM Failure handler
REM ============================================================

:fail
echo.
echo ============================================================
echo BUILD FAILED
echo ============================================================
echo.
echo Fix the reported issue and rerun this script.
echo Already-installed dependencies will be checked again.
echo.
pause
exit /b 1
