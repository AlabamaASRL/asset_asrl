
@echo off
setlocal EnableExtensions

rem ============================================================================
rem ASSET Windows build
rem
rem Requirements:
rem   - .venv at the repository root
rem   - LLVM/clang-cl 18.1.8
rem   - Visual Studio C++ Build Tools and Windows SDK
rem
rem The script installs Python build dependencies into .venv only.
rem ============================================================================

title ASSET Windows Build

echo.
echo ============================================================
echo ASSET Windows Build
echo ============================================================
echo.

rem Always run from the repository root, regardless of the caller's directory.
set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist "%ROOT%CMakeLists.txt" (
    echo ERROR: CMakeLists.txt was not found at:
    echo        "%ROOT%"
    echo Make sure this script is in the ASSET repository root.
    goto :FAIL
)

rem ---------------------------------------------------------------------------
rem 1. Activate the repository's virtual environment.
rem ---------------------------------------------------------------------------

echo [1/8] Checking the repository virtual environment...

set "VENV=%ROOT%.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "ACTIVATE=%VENV%\Scripts\activate.bat"

if not exist "%PYTHON%" (
    echo ERROR: Python was not found at:
    echo        "%PYTHON%"
    echo.
    echo Create the environment from the repository root with:
    echo   py -3.12 -m venv .venv
    goto :FAIL
)

if not exist "%ACTIVATE%" (
    echo ERROR: Virtual environment activation script is missing:
    echo        "%ACTIVATE%"
    goto :FAIL
)

call "%ACTIVATE%"
if errorlevel 1 (
    echo ERROR: Failed to activate .venv.
    goto :FAIL
)

rem Use the explicit venv interpreter for all Python operations.
"%PYTHON%" -c "import sys; print('Python:', sys.executable); print('Version:', sys.version.split()[0])"
if errorlevel 1 (
    echo ERROR: The venv Python interpreter could not run.
    goto :FAIL
)

rem ---------------------------------------------------------------------------
rem 2. Check pip.
rem ---------------------------------------------------------------------------

echo.
echo [2/8] Checking pip...

"%PYTHON%" -m pip --version
if errorlevel 1 (
    echo pip is unavailable. Attempting to bootstrap pip...
    "%PYTHON%" -m ensurepip --upgrade
    if errorlevel 1 (
        echo ERROR: Could not install pip in .venv.
        goto :FAIL
    )
)

rem ---------------------------------------------------------------------------
rem 3. Check Visual Studio C++ Build Tools.
rem ---------------------------------------------------------------------------

echo.
echo [3/8] Checking Visual Studio C++ Build Tools...

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINSTALL="

if not exist "%VSWHERE%" (
    echo ERROR: Visual Studio Installer vswhere.exe was not found.
    echo Install Visual Studio 2022 Build Tools with the C++ workload.
    echo Required components include MSVC x64/x86 tools and a Windows SDK.
    goto :FAIL
)

for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    if not defined VSINSTALL set "VSINSTALL=%%I"
)

if not defined VSINSTALL (
    echo ERROR: Visual Studio C++ build tools were not detected.
    echo Install Visual Studio 2022 Build Tools with:
    echo   Desktop development with C++
    goto :FAIL
)

set "VSDEVCMD=%VSINSTALL%\Common7\Tools\VsDevCmd.bat"

if not exist "%VSDEVCMD%" (
    echo ERROR: VsDevCmd.bat was not found:
    echo        "%VSDEVCMD%"
    goto :FAIL
)

call "%VSDEVCMD%" -arch=x64 -host_arch=x64
if errorlevel 1 (
    echo ERROR: Could not initialize the Visual Studio build environment.
    goto :FAIL
)

echo Visual Studio: "%VSINSTALL%"

rem ---------------------------------------------------------------------------
rem 4. Find LLVM. Install the pinned version if it is missing.
rem ---------------------------------------------------------------------------

echo.
echo [4/8] Checking LLVM...

set "LLVM_BIN="
set "LLVM_BIN_DEFAULT=%ProgramFiles%\LLVM\bin"

if exist "%LLVM_BIN_DEFAULT%\clang-cl.exe" (
    set "LLVM_BIN=%LLVM_BIN_DEFAULT%"
)

if not defined LLVM_BIN (
    for /f "delims=" %%I in ('where.exe clang-cl.exe 2^>nul') do (
        if not defined LLVM_BIN set "LLVM_BIN=%%~dpI"
    )
)

if not defined LLVM_BIN (
    where.exe winget >nul 2>&1
    if errorlevel 1 (
        echo ERROR: LLVM is missing and winget is unavailable.
        echo Install LLVM 18.1.8 and rerun this script.
        goto :FAIL
    )

    echo LLVM was not found. Installing LLVM 18.1.8...
    winget install --exact --id LLVM.LLVM --version 18.1.8 --accept-source-agreements --accept-package-agreements

    if errorlevel 1 (
        echo ERROR: LLVM installation failed.
        goto :FAIL
    )

    if exist "%LLVM_BIN_DEFAULT%\clang-cl.exe" (
        set "LLVM_BIN=%LLVM_BIN_DEFAULT%"
    )
)

if not defined LLVM_BIN (
    echo ERROR: clang-cl.exe could not be located after installation.
    goto :FAIL
)

set "CLANG_CL=%LLVM_BIN%\clang-cl.exe"
set "LLD_LINK=%LLVM_BIN%\lld-link.exe"

if not exist "%CLANG_CL%" (
    echo ERROR: clang-cl.exe was not found:
    echo        "%CLANG_CL%"
    goto :FAIL
)

if not exist "%LLD_LINK%" (
    echo ERROR: lld-link.exe was not found:
    echo        "%LLD_LINK%"
    goto :FAIL
)

"%CLANG_CL%" --version
if errorlevel 1 (
    echo ERROR: clang-cl failed to run.
    goto :FAIL
)

rem Make LLVM available to CMake and its compiler subprocesses.
set "PATH=%LLVM_BIN%;%PATH%"

rem ---------------------------------------------------------------------------
rem 5. Install/check CMake and Ninja inside .venv.
rem ---------------------------------------------------------------------------

echo.
echo [5/8] Checking CMake and Ninja...

if not exist "%VENV%\Scripts\cmake.exe" (
    echo Installing CMake into .venv...
    "%PYTHON%" -m pip install cmake
    if errorlevel 1 (
        echo ERROR: Could not install CMake into .venv.
        goto :FAIL
    )
)

if not exist "%VENV%\Scripts\ninja.exe" (
    echo Installing Ninja into .venv...
    "%PYTHON%" -m pip install ninja
    if errorlevel 1 (
        echo ERROR: Could not install Ninja into .venv.
        goto :FAIL
    )
)

set "CMAKE=%VENV%\Scripts\cmake.exe"
set "NINJA=%VENV%\Scripts\ninja.exe"

if not exist "%CMAKE%" (
    echo ERROR: CMake executable is missing from .venv.
    goto :FAIL
)

if not exist "%NINJA%" (
    echo ERROR: Ninja executable is missing from .venv.
    goto :FAIL
)

"%CMAKE%" --version
if errorlevel 1 (
    echo ERROR: CMake did not run successfully.
    goto :FAIL
)

"%NINJA%" --version
if errorlevel 1 (
    echo ERROR: Ninja did not run successfully.
    goto :FAIL
)

rem ---------------------------------------------------------------------------
rem 6. Check Intel MKL headers and libraries.
rem ---------------------------------------------------------------------------

echo.
echo [6/8] Checking Intel MKL...

call :FindMKL

if not defined MKL_INCLUDE_DIR (
    echo MKL headers or libraries are missing. Installing MKL packages...
    "%PYTHON%" -m pip install mkl-devel intel-openmp
    if errorlevel 1 (
        echo ERROR: Could not install mkl-devel and intel-openmp.
        goto :FAIL
    )

    call :FindMKL
)

if not defined MKL_INCLUDE_DIR (
    echo ERROR: mkl.h was not found inside .venv.
    echo Check the mkl-devel installation and package layout.
    goto :FAIL
)

if not defined MKL_INTERFACE (
    echo ERROR: mkl_intel_lp64_dll.lib was not found inside .venv.
    goto :FAIL
)

if not defined MKL_SEQTHR (
    echo ERROR: mkl_sequential_dll.lib was not found inside .venv.
    goto :FAIL
)

if not defined MKL_CORE (
    echo ERROR: mkl_core_dll.lib was not found inside .venv.
    goto :FAIL
)

echo MKL include directory:
echo   "%MKL_INCLUDE_DIR%"
echo MKL interface library:
echo   "%MKL_INTERFACE%"
echo MKL sequential library:
echo   "%MKL_SEQTHR%"
echo MKL core library:
echo   "%MKL_CORE%"

rem ---------------------------------------------------------------------------
rem 7. Configure the CMake build.
rem ---------------------------------------------------------------------------

echo.
echo [7/8] Configuring CMake...

set "BUILD_DIR=%ROOT%build-windows-venv"
set "MKLROOT=%VENV%"

if exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Existing CMake cache found:
    echo   "%BUILD_DIR%\CMakeCache.txt"
    echo.
    echo To avoid reusing a cache configured with another compiler,
    echo remove the build-windows-venv directory and rerun this script.
    echo The script will not delete an existing build automatically.
    goto :FAIL
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create the build directory.
    goto :FAIL
)

"%CMAKE%" -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPIP_INSTALL=True ^
    -DBUILD_SPHINX_DOCS=True ^
    "-DCMAKE_C_COMPILER=%CLANG_CL%" ^
    "-DCMAKE_CXX_COMPILER=%CLANG_CL%" ^
    "-DCMAKE_LINKER=%LLD_LINK%" ^
    "-DCMAKE_MAKE_PROGRAM=%NINJA%" ^
    "-DMKLROOT=%MKLROOT%" ^
    "-DMKL_INCLUDE_DIRS=%MKL_INCLUDE_DIR%" ^
    "-DMKL_INTERFACE_LIBRARY=%MKL_INTERFACE%" ^
    "-DMKL_SEQTHR_LIBRARY=%MKL_SEQTHR%" ^
    "-DMKL_CORE_LIBRARY=%MKL_CORE%" ^
    "-DMKL_LIBRARIES=%MKL_INTERFACE%;%MKL_SEQTHR%;%MKL_CORE%"

if errorlevel 1 (
    echo ERROR: CMake configuration failed.
    goto :FAIL
)

rem ---------------------------------------------------------------------------
rem 8. Build ASSET and the wheel target.
rem ---------------------------------------------------------------------------

echo.
echo [8/8] Building ASSET and pypiwheel...

"%CMAKE%" --build "%BUILD_DIR%" --target asset pypiwheel --parallel 2
if errorlevel 1 (
    echo ERROR: ASSET build failed.
    goto :FAIL
)

echo.
echo ============================================================
echo BUILD SUCCEEDED
echo ============================================================
echo Build directory:
echo   "%BUILD_DIR%"
echo.
echo Check the build output for generated wheel files.
echo.
pause
exit /b 0


rem ============================================================================
rem Search only the repository's .venv for the required MKL files.
rem ============================================================================

:FindMKL
set "MKL_INCLUDE_DIR="
set "MKL_INTERFACE="
set "MKL_SEQTHR="
set "MKL_CORE="

for /f "delims=" %%I in ('dir /s /b "%VENV%\mkl.h" 2^>nul') do (
    if not defined MKL_INCLUDE_DIR set "MKL_INCLUDE_DIR=%%~dpI"
)

for /f "delims=" %%I in ('dir /s /b "%VENV%\mkl_intel_lp64_dll.lib" 2^>nul') do (
    if not defined MKL_INTERFACE set "MKL_INTERFACE=%%I"
)

for /f "delims=" %%I in ('dir /s /b "%VENV%\mkl_sequential_dll.lib" 2^>nul') do (
    if not defined MKL_SEQTHR set "MKL_SEQTHR=%%I"
)

for /f "delims=" %%I in ('dir /s /b "%VENV%\mkl_core_dll.lib" 2^>nul') do (
    if not defined MKL_CORE set "MKL_CORE=%%I"
)

exit /b 0


:FAIL
echo.
echo ============================================================
echo BUILD FAILED
echo ============================================================
echo.
echo Fix the reported issue and rerun this script.
echo The script uses the repository's .venv and does not
echo automatically delete an existing build directory.
echo.
pause
exit /b 1
