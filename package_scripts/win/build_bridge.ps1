# MNNPyBridge
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
    [Parameter(Mandatory=$true)][String]$version,
    [Parameter(Mandatory=$true)][String]$pyc_env,
    [Parameter(Mandatory=$true)][String]$mnn_path,
    [Parameter(Mandatory=$true)][String]$path,
    [Switch]$x86
)

# build process may failed because of lnk1181, but be success when run again
# Run expr, return if success, otherwise try again until try_times
function Retry([String]$expr, [Int]$try_times) {
  $cnt = 0
  do {
   $cnt++
   try {
     Invoke-Expression $expr
     return
   } catch { }
 } while($cnt -lt $try_times)
 throw "Failed: $expr"
}

$erroractionpreference = "stop"
$PACKAGE_PATH = $(Resolve-Path $path).Path
$PACKAGE_LIB_PATH = "$PACKAGE_PATH\lib"
if ($x86) {
    $PACKAGE_LIB_PATH = "$PACKAGE_LIB_PATH\x86"
} else {
    $PACKAGE_LIB_PATH = "$PACKAGE_LIB_PATH\x64"
}
$MNN_PACKAGE_PATH = $(Resolve-Path $mnn_path).Path

pushd pymnn\3rd_party
Remove-Item MNN -Recurse -ErrorAction Ignore
mkdir -p MNN\lib
cp -r $MNN_PACKAGE_PATH\* MNN\lib
cp -r ..\..\include MNN
popd

#clear and create package directory
powershell ./schema/generate.ps1
pushd $PACKAGE_PATH
Remove-Item include -Recurse -ErrorAction Ignore
Remove-Item wrapper -Recurse -ErrorAction Ignore
mkdir -p include
mkdir -p wrapper
mkdir -p $PACKAGE_LIB_PATH\Debug\MD -ErrorAction SilentlyContinue
mkdir -p $PACKAGE_LIB_PATH\Debug\MT -ErrorAction SilentlyContinue
mkdir -p $PACKAGE_LIB_PATH\Debug\Static -ErrorAction SilentlyContinue
mkdir -p $PACKAGE_LIB_PATH\Release\MD -ErrorAction SilentlyContinue
mkdir -p $PACKAGE_LIB_PATH\Release\MT -ErrorAction SilentlyContinue
mkdir -p $PACKAGE_LIB_PATH\Release\Static -ErrorAction SilentlyContinue
popd

# assume $PACKAGE_PATH exist
cp pymnn\src\MNNPyBridge.h $PACKAGE_PATH\include
Remove-Item pymnn_pyc_tmp -Recurse -ErrorAction Ignore
mkdir pymnn_pyc_tmp
cp -r pymnn\pip_package\MNN pymnn_pyc_tmp
pushd pymnn_pyc_tmp
Remove-Item MNN -Include __pycache__ -Recurse
pushd MNN
rm -r -force tools
(Get-Content __init__.py).replace('from . import tools', '') | Set-Content __init__.py
popd
popd
conda activate $pyc_env
python -c "import compileall; compileall.compile_dir('./pymnn_pyc_tmp', force=True)"
conda deactivate
Remove-Item .\pymnn_pyc_tmp -Include *.py -Recurse
Set-Content -Path pymnn_pyc_tmp\version.py -Value "__version__ = '$version'"
cp -r .\pymnn_pyc_tmp\* $PACKAGE_PATH\wrapper -Force
rm -r -force pymnn_pyc_tmp

$CMAKE_ARGS = "-DPYMNN_USE_ALINNPYTHON=ON -DPYMNN_RUNTIME_CHECK_VM=ON -DPYMNN_EXPR_API=ON -DPYMNN_NUMPY_USABLE=ON -DPYMNN_TRAIN_API=ON"

Remove-Item pymnn_build -Recurse -ErrorAction Ignore
mkdir pymnn_build
pushd pymnn_build

##### Debug/MT ####
#Remove-Item CMakeCache.txt -ErrorAction Ignore
#Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_WIN_RUNTIME_MT=ON ../pymnn"
#Retry "ninja" 2
#cp mnnpybridge.lib $PACKAGE_LIB_PATH\Debug\MT
#cp mnnpybridge.dll $PACKAGE_LIB_PATH\Debug\MT
#cp mnnpybridge.pdb $PACKAGE_LIB_PATH\Debug\MT
#rm mnnpybridge.*

##### Debug/MD ####
#Remove-Item CMakeCache.txt -ErrorAction Ignore
#Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_WIN_RUNTIME_MT=OFF ../pymnn"
#Retry "ninja" 2
#cp mnnpybridge.lib $PACKAGE_LIB_PATH\Debug\MD
#cp mnnpybridge.dll $PACKAGE_LIB_PATH\Debug\MD
#cp mnnpybridge.pdb $PACKAGE_LIB_PATH\Debug\MD
#rm mnnpybridge.*

##### Debug/Static ####
#Remove-Item CMakeCache.txt -ErrorAction Ignore
#Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_WIN_RUNTIME_MT=OFF -DMNN_BUILD_SHARED_LIBS=OFF ../pymnn"
#Retry "ninja" 2
#cp mnnpybridge.lib $PACKAGE_LIB_PATH\Debug\Static
#rm mnnpybridge.*

##### Release/MT ####
#Remove-Item CMakeCache.txt -ErrorAction Ignore
#Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON ../pymnn"
#Retry "ninja" 2
#cp mnnpybridge.lib $PACKAGE_LIB_PATH\Release\MT
#cp mnnpybridge.dll $PACKAGE_LIB_PATH\Release\MT
#cp mnnpybridge.pdb $PACKAGE_LIB_PATH\Release\MT
#rm mnnpybridge.*

##### Release/MD ####
Remove-Item CMakeCache.txt -ErrorAction Ignore
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=OFF ../pymnn"
Retry "ninja" 2
cp mnnpybridge.lib $PACKAGE_LIB_PATH\Release\MD
cp mnnpybridge.dll $PACKAGE_LIB_PATH\Release\MD
cp mnnpybridge.pdb $PACKAGE_LIB_PATH\Release\MD
rm mnnpybridge.*

##### Release/Static ####
#Remove-Item CMakeCache.txt -ErrorAction Ignore
#Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=OFF -DMNN_BUILD_SHARED_LIBS=OFF ../pymnn"
#Retry "ninja" 2
#cp mnnpybridge.lib $PACKAGE_LIB_PATH\Release\Static

popd