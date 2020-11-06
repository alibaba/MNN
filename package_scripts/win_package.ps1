# MNN
#  |-- Debug
#  |     |--- MD
#  |     |--- MT
#  |-- Release
#        |--- MD
#        |--- MT

$erroractionpreference = "stop"

Set-Variable -Name WINDOWS_PACKAGE_NAME -Value "MNN"

#clear and create package directory
powershell ./schema/generate.ps1
Set-Variable -Name WINDOWS_PACKAGE_PATH -Value "$(pwd)\$WINDOWS_PACKAGE_NAME"
Remove-Item $WINDOWS_PACKAGE_PATH -Recurse -ErrorAction Ignore
mkdir $WINDOWS_PACKAGE_PATH\
cd $WINDOWS_PACKAGE_PATH
mkdir -p Debug\MD
mkdir -p Debug\MT
mkdir -p Release\MD
mkdir -p Release\MT
cd ..

Remove-Item build -Recurse -ErrorAction Ignore
mkdir build
pushd build
# tools without dependency, static library without sep_build
#cmake -G "Ninja" -DMNN_SEP_BUILD=OFF -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_CONVERTER=ON -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON ..
#ninja
#pushd $WINDOWS_PACKAGE_PATH
#cp ..\build\*.exe MNN_Windows_tools
#cp ..\build\*.pdb MNN_Windows_tools
#cp ..\build\MNN.lib MNN_Windows_lib\Static_Library
#popd

Remove-Item CMakeCache.txt -ErrorAction Ignore
cmake -G "Ninja" -DMNN_SEP_BUILD=OFF -DCMAKE_BUILD_TYPE=Debug -DMNN_WIN_RUNTIME_MT=ON -DMNN_OPENCL=ON ..
ninja
cp MNN.lib $WINDOWS_PACKAGE_PATH\Debug\MT
cp MNN.dll $WINDOWS_PACKAGE_PATH\Debug\MT
cp MNN.pdb $WINDOWS_PACKAGE_PATH\Debug\MT

Remove-Item CMakeCache.txt -ErrorAction Ignore
cmake -G "Ninja" -DMNN_SEP_BUILD=OFF -DCMAKE_BUILD_TYPE=Debug -DMNN_WIN_RUNTIME_MT=OFF -DMNN_OPENCL=ON ..
ninja
cp MNN.lib $WINDOWS_PACKAGE_PATH\Debug\MD
cp MNN.dll $WINDOWS_PACKAGE_PATH\Debug\MD
cp MNN.pdb $WINDOWS_PACKAGE_PATH\Debug\MD

Remove-Item CMakeCache.txt -ErrorAction Ignore
cmake -G "Ninja" -DMNN_SEP_BUILD=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON -DMNN_OPENCL=ON ..
ninja
cp MNN.lib $WINDOWS_PACKAGE_PATH\Release\MT
cp MNN.dll $WINDOWS_PACKAGE_PATH\Release\MT
cp MNN.pdb $WINDOWS_PACKAGE_PATH\Release\MT

Remove-Item CMakeCache.txt -ErrorAction Ignore
cmake -G "Ninja" -DMNN_SEP_BUILD=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=OFF -DMNN_OPENCL=ON ..
ninja
cp MNN.lib $WINDOWS_PACKAGE_PATH\Release\MD
cp MNN.dll $WINDOWS_PACKAGE_PATH\Release\MD
cp MNN.pdb $WINDOWS_PACKAGE_PATH\Release\MD

popd