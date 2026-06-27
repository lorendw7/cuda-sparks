@echo off
REM ---------------------------------------------------------------------------
REM  cuda-sparks build helper (Ninja + MSVC + nvcc)
REM    build.bat            - configure (first time) + build everything
REM    build.bat run1       - build, then run Phase 1 (CPU)
REM    build.bat run2       - build, then run Phase 2 (CUDA)
REM    build.bat clean      - delete the build/ folder and start fresh
REM
REM  We use the Ninja generator (bundled with Visual Studio) instead of the
REM  Visual Studio generator, because VS 2026 does not ship the CUDA toolset
REM  integration. Ninja calls nvcc directly, so no admin install is needed.
REM ---------------------------------------------------------------------------
setlocal
set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set NINJA_DIR=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set ROOT=%ROOT:~0,-1%
set BUILD_DIR=%ROOT%\build

if "%1"=="clean" (
    echo Removing build directory...
    rmdir /s /q "%BUILD_DIR%" 2>nul
    exit /b 0
)

REM Bring MSVC (cl.exe) and the Windows SDK into PATH, then add Ninja.
call %VCVARS% >nul 2>&1
set PATH=%NINJA_DIR%;%PATH%

REM Configure once (creates build/, downloads GLFW + glad the first time).
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Configuring ^(first build downloads GLFW + glad^)...
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B "%BUILD_DIR%" -S "%ROOT%"
    if errorlevel 1 exit /b 1
)

echo Building...
cmake --build "%BUILD_DIR%"
if errorlevel 1 exit /b 1

if "%1"=="run1" (
    echo.
    "%BUILD_DIR%\phases\phase1_cpu_baseline\phase1.exe"
)
if "%1"=="run2" (
    echo.
    "%BUILD_DIR%\phases\phase2_cuda_migration\phase2.exe"
)

endlocal
