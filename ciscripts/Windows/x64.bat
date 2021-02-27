call "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars64.bat"

echo "GENERATING"
Powershell.exe -executionpolicy remotesigned -File .\schema\generate.ps1
echo "GENERATED"

mkdir winbuild
cd winbuild

cmake ../ -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DMNN_SEP_BUILD=0 -DMNN_USE_SYSTEM_LIB=ON -DMNN_BUILD_TEST=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TOOLS=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_TRAIN=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON
ninja
