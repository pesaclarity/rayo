@echo off
rem The C library, its tests and the quickstart on Windows with MSVC (cl) or
rem clang-cl, in a Visual Studio x64 developer environment found by vswhere.
rem C89 itself is checked by gcc and clang -std=c89 -pedantic (lint.sh); here
rem the Win32 layer (MapViewOfFile, Win32 threads, the CPU's cores) is built
rem and tested with Microsoft's toolchain, with every warning an error.
rem
rem   tool\ci\windows.cmd cl
rem   tool\ci\windows.cmd clang-cl
setlocal enabledelayedexpansion
cd /d "%~dp0\..\.."
set CC=%1
if "%CC%"=="" set CC=cl

set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VS=%%i
if not defined VS (echo windows.cmd: no Visual Studio with the C tools & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

rem MSVC: C (/TC), warnings as errors, no secure-CRT nags for C89's strcpy
rem and fopen. MSVC does not contract a * b + c into an FMA under its default
rem /fp:precise, so the floats are every other build's. clang-cl: the same,
rem as C89, with -ffp-contract=off.
set FLAGS=/nologo /TC /O2 /W3 /WX /D_CRT_SECURE_NO_WARNINGS /Iinclude /Isrc
if /i "%CC%"=="clang-cl" set FLAGS=%FLAGS% /clang:-std=c89 /clang:-ffp-contract=off

set SOURCES=
for /r src %%f in (*.c) do set SOURCES=!SOURCES! "%%f"
set OUT=%RUNNER_TEMP%
if not defined OUT set OUT=%TEMP%
set OUT=%OUT%\rayo-%CC%
if exist "%OUT%" rmdir /s /q "%OUT%"
mkdir "%OUT%" || exit /b 1

echo == %CC%: the C tests
%CC% %FLAGS% /Itests %SOURCES% tests\*.c /Fo"%OUT%\\" /Fe"%OUT%\tests.exe" || exit /b 1
for %%k in (generic simd best) do (
  echo -- kernels: %%k
  set RAYO_TEST_KERNELS=%%k
  "%OUT%\tests.exe" || exit /b 1
)
set RAYO_TEST_KERNELS=

echo == %CC%: examples\c\quickstart.c
del /q "%OUT%\*.obj" >nul 2>&1
%CC% %FLAGS% %SOURCES% examples\c\quickstart.c /Fo"%OUT%\\" /Fe"%OUT%\quickstart.exe" || exit /b 1
"%OUT%\quickstart.exe" examples\classifier\classifier.weights > "%OUT%\quickstart.txt" || exit /b 1
type "%OUT%\quickstart.txt"
rem The probabilities NumPy gives for ids 4 17 9 (examples/classifier/expected.json).
powershell -NoProfile -Command "if (((Get-Content '%OUT%\quickstart.txt') -join ',') -ne '0.275287,0.227712,0.497001') { exit 1 }" || (echo quickstart: unexpected output & exit /b 1)
echo windows: ok
