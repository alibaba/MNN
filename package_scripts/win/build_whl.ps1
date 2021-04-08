Param(
    [Parameter(Mandatory=$true)][String]$version,
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
$ARGS = "--version $version"
if ($x86) {
    $ARGS = "$ARGS --x86"
}

powershell ./schema/generate.ps1

Remove-Item pymnn_build -Recurse -ErrorAction Ignore
mkdir pymnn_build
pushd pymnn_build
$CMAKE_ARGS = "-DMNN_SEP_BUILD=OFF -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON "
if ($opencl) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_OPENCL=ON"
}
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS .."
ninja MNN MNNTrain MNNConvert
popd

pushd pymnn/pip_package
Set-Content -Path MNN/version.py -Value "__version__ = '$version'"
Remove-Item dist -Recurse -ErrorAction Ignore
Remove-Item build -Recurse -ErrorAction Ignore
mkdir dist
mkdir build

Foreach ($env in $python_versions) {
    Invoke-Expression "pyenv global $env"
    Invoke-Expression "python build_wheel.py $ARGS"
}
cp dist/* $PACKAGE_PATH
Remove-Item MNN/version.py -ErrorAction Ignore
popd