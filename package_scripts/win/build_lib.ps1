# MNN
#  |-- Debug
#  |     |--- MD
#  |     |--- MT
#  |     |--- Static
#  |
#  |-- Release
#        |--- MD
#        |--- MT
#        |--- Static

Param(
    [Parameter(Mandatory=$true)][String]$path,
    [Switch]$opencl
)
$erroractionpreference = "stop"
Remove-Item $path -Recurse -ErrorAction Ignore
mkdir -p $path
$PACKAGE_PATH = $(Resolve-Path $path).Path

#clear and create package directory
powershell ./schema/generate.ps1
pushd $PACKAGE_PATH
mkdir -p Debug\MD
mkdir -p Debug\MT
mkdir -p Debug\Static
mkdir -p Release\MD
mkdir -p Release\MT
mkdir -p Release\Static
popd

$CMAKE_ARGS = "-DMNN_SEP_BUILD=OFF -DMNN_BUILD_TRAIN=ON"
if ($opencl) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_OPENCL=ON"
}

#Remove-Item build -Recurse -ErrorAction Ignore
mkdir build
pushd build

##### Debug/MT ####
Remove-Item CMakeCache.txt -ErrorAction Ignore
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_WIN_RUNTIME_MT=ON .."
ninja
cp MNN.lib $PACKAGE_PATH\Debug\MT
cp MNN.dll $PACKAGE_PATH\Debug\MT
cp MNN.pdb $PACKAGE_PATH\Debug\MT
rm MNN.*

##### Debug/MD ####
Remove-Item CMakeCache.txt -ErrorAction Ignore
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_WIN_RUNTIME_MT=OFF .."
ninja
cp MNN.lib $PACKAGE_PATH\Debug\MD
cp MNN.dll $PACKAGE_PATH\Debug\MD
cp MNN.pdb $PACKAGE_PATH\Debug\MD
rm MNN.*

##### Debug/Static ####
Remove-Item CMakeCache.txt -ErrorAction Ignore
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_WIN_RUNTIME_MT=OFF -DMNN_BUILD_SHARED_LIBS=OFF .."
ninja
cp MNN.lib $PACKAGE_PATH\Debug\Static
rm MNN.*

##### Release/MT ####
Remove-Item CMakeCache.txt -ErrorAction Ignore
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON .."
ninja
cp MNN.lib $PACKAGE_PATH\Release\MT
cp MNN.dll $PACKAGE_PATH\Release\MT
cp MNN.pdb $PACKAGE_PATH\Release\MT
rm MNN.*

##### Release/MD ####
Remove-Item CMakeCache.txt -ErrorAction Ignore
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=OFF .."
ninja
cp MNN.lib $PACKAGE_PATH\Release\MD
cp MNN.dll $PACKAGE_PATH\Release\MD
cp MNN.pdb $PACKAGE_PATH\Release\MD
rm MNN.*

##### Release/Static ####
Remove-Item CMakeCache.txt -ErrorAction Ignore
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=OFF -DMNN_BUILD_SHARED_LIBS=OFF .."
ninja
cp MNN.lib $PACKAGE_PATH\Release\Static

popd