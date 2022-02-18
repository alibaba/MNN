Param(
    [Parameter(Mandatory=$true)][String]$version,
    [Parameter(Mandatory=$true)][String]$path,
    [String]$pyenvs,
    [String]$backends,
    [Switch]$x86
)

# build it according to cmake_cmd, exit 1 when any error occur
function Build([String]$cmake_cmd, [String]$ninja_cmd = "ninja") {
    Invoke-Expression $cmake_cmd
    # build process may failed because of lnk1181, but be success when run again
    $try_times = 2
    if ($LastExitCode -eq 0) {
        For ($cnt = 0; $cnt -lt $try_times; $cnt++) {
            try {
                Invoke-Expression $ninja_cmd
                if ($LastExitCode -eq 0) {
                    return
                }
            } catch {}
        }
    }
    exit 1
}

$erroractionpreference = "stop"
$python_versions = $pyenvs.Split(",")

New-Item -Path $path -ItemType Directory -ErrorAction Ignore
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
$CMAKE_ARGS = "-DMNN_SEP_BUILD=OFF -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_AVX512=ON"
if ($backends -ne $null) {
    Foreach($backend in $backends.Split(",")) {
        if ($backend -eq "opencl") {
            $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_OPENCL=ON"
        } elseif ($backend -eq "vulkan") {
            $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_VULKAN=ON"
        }
    }
}
Build "cmake -G Ninja $CMAKE_ARGS .." "ninja MNN MNNTrain MNNConvert MNNOpenCV"
popd

pushd pymnn/pip_package
Set-Content -Path MNN/version.py -Value "__version__ = '$version'"
Remove-Item dist -Recurse -ErrorAction Ignore
Remove-Item build -Recurse -ErrorAction Ignore
mkdir dist
mkdir build

if ($pyenvs -eq $null) {
    Invoke-Expression "python build_wheel.py $ARGS"
} else {
    Foreach ($env in $pyenvs.Split(",")) {
        Invoke-Expression "conda activate $env"
        Invoke-Expression "python build_wheel.py $ARGS"
        conda deactivate
        if ($LastExitCode -ne 0) {
            exit 1
        }
    }
}

cp dist/* $PACKAGE_PATH
Remove-Item MNN/version.py -ErrorAction Ignore
popd

if ($x86) {
    $env:CONDA_FORCE_32BIT=""
}