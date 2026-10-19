@echo off

if "%MSVC_VERSION%" == "10" goto msvc10

call "C:\Program Files (x86)\Microsoft Visual Studio %MSVC_VERSION%.0\VC\vcvarsall.bat" amd64
goto build

:msvc10
call "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x64

:build
nmake

