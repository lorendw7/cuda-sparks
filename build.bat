@echo off
REM ---------------------------------------------------------------------------
REM  cuda-sparks build helper
REM    build.bat            - configure (first time) + build everything (Release)
REM    build.bat run        - build, then run Phase 1
REM    build.bat clean      - delete the build/ folder and start fresh
REM ---------------------------------------------------------------------------
setlocal
set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set ROOT=%~dp0
set BUILD_DIR=%ROOT%build

call %VCVARS% >nul 2>&1

if "%1"=="clean" (
    echo Removing build directory...
    rmdir /s /q "%BUILD_DIR%" 2>nul
    exit /b 0
)

REM Configure once (creates the build/ folder + downloads GLFW the first time).
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo Configuring ^(first build downloads GLFW + glad^)...
    cmake -B "%BUILD_DIR%" -S "%ROOT%" -DCMAKE_BUILD_TYPE=Release
    if errorlevel 1 exit /b 1
)

echo Building...
cmake --build "%BUILD_DIR%" --config Release --parallel
if errorlevel 1 exit /b 1

if "%1"=="run" (
    echo.
    "%BUILD_DIR%\phases\phase1_cpu_baseline\Release\phase1.exe"
)

endlocal
