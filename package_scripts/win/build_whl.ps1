Param(
    [Parameter(Mandatory=$true)][String]$version,
    [Parameter(Mandatory=$true)][String]$path,
    [String]$pyenvs,
    [String]$backends,
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
$python_versions = $pyenvs.Split(",")

Remove-Item $path -Recurse -ErrorAction Ignore
mkdir -p $path
$PACKAGE_PATH = $(Resolve-Path $path).Path
$ARGS = "--version $version"
if ($x86) {
    $ARGS = "$ARGS --x86"
    $env:CONDA_FORCE_32BIT=1
}

powershell ./schema/generate.ps1

Remove-Item pymnn_build -Recurse -ErrorAction Ignore
mkdir pymnn_build
pushd pymnn_build
$CMAKE_ARGS = "-DMNN_SEP_BUILD=OFF -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON "
if ($backends -ne $null) {
    Foreach($backend in $backends.Split(",")) {
        if ($backend -eq "opencl") {
            $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_OPENCL=ON"
        } elseif ($backend -eq "vulkan") {
            $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_VULKAN=ON"
        }
    }
}
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS .."
Retry "ninja MNN MNNTrain MNNConvert" 2
popd

pushd pymnn/pip_package
Set-Content -Path MNN/version.py -Value "__version__ = '$version'"
Remove-Item dist -Recurse -ErrorAction Ignore
Remove-Item build -Recurse -ErrorAction Ignore
mkdir dist
mkdir build

if ($pyenvs -eq $null) {
    Retry "python build_wheel.py $ARGS" 2
} else {
    Foreach ($env in $pyenvs.Split(",")) {
        Invoke-Expression "conda activate $env"
        Retry "python build_wheel.py $ARGS" 2
        Invoke-Expression "conda deactivate"
    }
}

cp dist/* $PACKAGE_PATH
Remove-Item MNN/version.py -ErrorAction Ignore
popd

if ($x86) {
    $env:CONDA_FORCE_32BIT=""
}