#   MNN_Windows
#     |------- MNN_Windows_lib
#                   |---------- Dynamic_Library
#                   |---------- Static_Library
#     |------- MNN_Windows_tools

$erroractionpreference = "stop"

Set-Variable -Name WINDOWS_PACKAGE_NAME -Value "MNN_Windows"

#clear and create package directory
powershell ./schema/generate.ps1
Set-Variable -Name WINDOWS_PACKAGE_PATH -Value "$(pwd)\$WINDOWS_PACKAGE_NAME"
Remove-Item $WINDOWS_PACKAGE_PATH -Recurse -ErrorAction Ignore
mkdir $WINDOWS_PACKAGE_PATH\
cd $WINDOWS_PACKAGE_PATH
mkdir -p MNN_Windows_lib\Dynamic_Library
mkdir -p MNN_Windows_lib\Static_Library
mkdir MNN_Windows_tools
cd ..

Remove-Item build -Recurse -ErrorAction Ignore
mkdir build
cd build
# tools without dependency, static library without sep_build
cmake -G "Ninja" -DMNN_SEP_BUILD=OFF -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_CONVERTER=ON -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON ..
ninja
pushd $WINDOWS_PACKAGE_PATH
cp ..\build\*.exe MNN_Windows_tools
cp ..\build\*.pdb MNN_Windows_tools
cp ..\build\MNN.lib MNN_Windows_lib\Static_Library
popd

#dynamic library without sep_build
rm .\CMakeCache.txt
cmake -G "Ninja" -DMNN_SEP_BUILD=OFF ..
ninja
cd $WINDOWS_PACKAGE_PATH
cp ..\build\MNN.lib MNN_Windows_lib\Dynamic_Library
cp ..\build\MNN.dll MNN_Windows_lib\Dynamic_Library
cp ..\build\MNN.pdb MNN_Windows_lib\Dynamic_Library

# Compress MNN_Windows_lib and MNN_Windows_tools
Compress-Archive -Path MNN_Windows_lib -DestinationPath MNN_Windows_lib.zip -Update -CompressionLevel Optimal
Compress-Archive -Path MNN_Windows_tools -DestinationPath MNN_Windows_tools.zip -Update -CompressionLevel Optimal