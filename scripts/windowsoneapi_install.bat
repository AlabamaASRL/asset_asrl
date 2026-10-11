@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM SPDX-FileCopyrightText: 2022 Intel Corporation
REM
REM SPDX-License-Identifier: MIT

if "%~1"=="" (
  echo Usage: %~nx0 ^<basekit_url^> [components]
  exit /b 2
)

set "URL=%~1"
set "COMPONENTS=%~2"
set "INSTALL_DIR=%TEMP%\oneapi_install"
set "WEBIMAGE=%TEMP%\webimage.exe"
set "EXTRACT_DIR=webimage_extracted"
set "EXTRACT_LOG=extract.log"
set "BOOTSTRAP_LOG=bootstrapper.log"

echo [INFO] Starting oneAPI installer
echo [INFO] URL: %URL%
if defined COMPONENTS (
  echo [INFO] Components: %COMPONENTS%
) else (
  echo [INFO] Components: ^<default^>
)

if exist "%EXTRACT_DIR%" (
  echo [INFO] Removing stale extract directory: %EXTRACT_DIR%
  rd /s/q "%EXTRACT_DIR%"
  if exist "%EXTRACT_DIR%" (
    echo [ERROR] Failed to remove stale extract directory.
    exit /b 3
  )
)

echo [INFO] Downloading installer to %WEBIMAGE%
curl.exe --fail --location --output "%WEBIMAGE%" --url "%URL%" --retry 5 --retry-delay 5
if errorlevel 1 (
  echo [ERROR] Failed to download oneAPI installer.
  exit /b 10
)

echo [INFO] Extracting installer to %EXTRACT_DIR%
start /b /wait "%WEBIMAGE%" -s -x -f "%EXTRACT_DIR%" --log "%EXTRACT_LOG%"
set "EXTRACT_EXIT=%ERRORLEVEL%"
if not "%EXTRACT_EXIT%"=="0" (
  echo [ERROR] Installer extraction failed with exit code %EXTRACT_EXIT%
  if exist "%EXTRACT_LOG%" type "%EXTRACT_LOG%"
  del /f /q "%WEBIMAGE%" >nul 2>&1
  if exist "%EXTRACT_DIR%" rd /s/q "%EXTRACT_DIR%"
  exit /b %EXTRACT_EXIT%
)

del /f /q "%WEBIMAGE%" >nul 2>&1

if not exist "%EXTRACT_DIR%\bootstrapper.exe" (
  echo [ERROR] bootstrapper.exe not found in %EXTRACT_DIR%
  if exist "%EXTRACT_DIR%" dir /s /b "%EXTRACT_DIR%"
  if exist "%EXTRACT_LOG%" type "%EXTRACT_LOG%"
  exit /b 11
)

echo [INFO] Running bootstrapper
if defined COMPONENTS (
  "%EXTRACT_DIR%\bootstrapper.exe" -s --action install --components=%COMPONENTS% --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=0 --log-dir=.
) else (
  "%EXTRACT_DIR%\bootstrapper.exe" -s --action install --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 -p=NEED_VS2022_INTEGRATION=0 --log-dir=.
)

set "INSTALL_EXIT=%ERRORLEVEL%"
echo [INFO] Bootstrapper exit code: %INSTALL_EXIT%

if exist "%EXTRACT_LOG%" (
  echo [INFO] Extract log:
  type "%EXTRACT_LOG%"
)

if exist "%BOOTSTRAP_LOG%" (
  echo [INFO] Bootstrapper log:
  type "%BOOTSTRAP_LOG%"
)

if exist "%EXTRACT_DIR%" (
  echo [INFO] Cleaning up %EXTRACT_DIR%
  rd /s/q "%EXTRACT_DIR%"
)

exit /b %INSTALL_EXIT%
