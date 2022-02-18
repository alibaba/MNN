if %1 EQU x86 (
    @call "%vs_env_setup%/vcvarsamd64_x86.bat"
    powershell "%~dp0test.ps1" -gpu -x86
) else (
    @call "%vs_env_setup%/vcvars64.bat"
    powershell "%~dp0test.ps1" -gpu
)