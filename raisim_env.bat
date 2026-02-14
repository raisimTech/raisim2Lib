@echo off
set "RAISIM_ROOT=%~dp0"
if "%RAISIM_ROOT:~-1%"=="\" set "RAISIM_ROOT=%RAISIM_ROOT:~0,-1%"

if not defined RAISIM_OS set "RAISIM_OS=win32"

set "RAISIM_BASE=%RAISIM_ROOT%\raisim\%RAISIM_OS%"
if not exist "%RAISIM_BASE%\bin\" set "RAISIM_BASE=%RAISIM_ROOT%\raisim"

set "RAYRAI_BASE=%RAISIM_ROOT%\rayrai\%RAISIM_OS%"
if not exist "%RAYRAI_BASE%\bin\" set "RAYRAI_BASE=%RAISIM_ROOT%\rayrai"

call :AddPath "%RAISIM_BASE%\bin"
call :AddPath "%RAYRAI_BASE%\bin"
call :AddPath "%RAISIM_BASE%\lib"
call :AddPath "%RAYRAI_BASE%\lib"

REM If you're using vcpkg to supply runtime DLLs (e.g. SDL2), add its bin folders too.
if not defined VCPKG_ROOT (
  if exist "C:\vcpkg\vcpkg.exe" set "VCPKG_ROOT=C:\vcpkg"
)
if defined VCPKG_ROOT (
  call :AddPath "%VCPKG_ROOT%\installed\x64-windows\bin"
  call :AddPath "%VCPKG_ROOT%\installed\x64-windows\debug\bin"
)
goto :eof

:AddPath
set "entry=%~1"
if "%entry%"=="" goto :eof
if not exist "%entry%\" goto :eof
echo;%PATH%; | find /I ";%entry%;" >nul
if not errorlevel 1 goto :eof
if "%PATH%"=="" (
  set "PATH=%entry%"
) else (
  set "PATH=%entry%;%PATH%"
)
echo Added to PATH: %entry%
goto :eof
