# .\package_scripts\win_pymm_package.ps1 -path MNN-CPU/py_whl/x64 -pyenvs "2.7.17,3.5.4,2.6.8,3.7.7,3.8.2"
# .\package_scripts\win_pymm_package.ps1 -x86 -path MNN-CPU/py_whl/x86 -pyenvs "2.7.17-win32,3.5.4-win32,2.6.8-win32,3.7.7-win32,3.8.2-win32"
# .\package_scripts\win_pymm_package.ps1 -path MNN-CPU-OPENCL/py_whl/x64 -pyenvs "2.7.17,3.5.4,2.6.8,3.7.7,3.8.2"
# .\package_scripts\win_pymm_package.ps1 -x86 -path MNN-CPU-OPENCL/py_whl/x86 -pyenvs "2.7.17-win32,3.5.4-win32,2.6.8-win32,3.7.7-win32,3.8.2-win32"
Param(
    [Parameter(Mandatory=$true)][String]$pyenvs,
    [Parameter(Mandatory=$true)][String]$path,
    [Switch]$x86,
    [Switch]$opencl
)

$erroractionpreference = "stop"
$python_versions = $pyenvs.Split(",")

Remove-Item $path -Recurse -ErrorAction Ignore
mkdir -p $path
$PACKAGE_PATH = $(Resolve-Path $path).Path
$ARGS = ""
if ($x86) {
    $ARGS = "--x86"
}

powershell ./schema/generate.ps1

Remove-Item pymnn_build -Recurse -ErrorAction Ignore
mkdir pymnn_build

pushd pymnn/pip_package
Remove-Item dist -Recurse -ErrorAction Ignore
mkdir dist

pushd pymnn_build
$CMAKE_ARGS = "-DMNN_SEP_BUILD=OFF -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON "
if ($opencl) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_OPENCL=ON"
}
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS .."
ninja MNN MNNTrain MNNConvert
popd

Foreach ($env in $python_versions) {
    pyenv global $env
    python build_wheel.py $ARGS
}
cp dist/* $PACKAGE_PATH
popd