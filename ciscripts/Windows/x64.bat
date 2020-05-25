call "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars64.bat"

echo "GENERATING"
Powershell.exe -executionpolicy remotesigned -File .\schema\generate.ps1
echo "GENERATED"

mkdir winbuild
cd winbuild

cmake ../ -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DMNN_SEP_BUILD=0
ninja
