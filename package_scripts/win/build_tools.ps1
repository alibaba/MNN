Param(
    [Parameter(Mandatory=$true)][String]$path,
    [Switch]$dynamic_link,
    [String]$backends,
    [Switch]$build_all,
    [Switch]$build_train, # MNN_BUILD_TRAIN
    [Switch]$build_tools, # MNN_BUILD_TOOLS
    [Switch]$build_quantools, # MNN_BUILD_QUANTOOLS
    [Switch]$build_evaluation, # MNN_EVALUATION
    [Switch]$build_converter, # MNN_BUILD_CONVERTER
    [Switch]$build_benchmark, # MNN_BUILD_BENCHMARK
    [Switch]$build_test, # MNN_BUILD_TEST
    [Switch]$build_demo # MNN_BUILD_DEMO
)

if ($build_all) {
    $build_train = $true
    $build_tools = $true
    $build_quantools = $true
    $build_evaluation = $true
    $build_converter = $true
    $build_benchmark = $true
    $build_test = $true
    $build_demo = $true
}

$erroractionpreference = "stop"
Remove-Item $path -Recurse -ErrorAction Ignore
mkdir -p $path
$TOOLS_PATH = $(Resolve-Path $path).Path

powershell ./schema/generate.ps1

$CMAKE_ARGS = "-DMNN_SEP_BUILD=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_AVX512=ON"
if ($dynamic_link) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_BUILD_SHARED_LIBS=ON"
} else {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_WIN_RUNTIME_MT=ON"
}
if ($build_train) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_BUILD_TRAIN=ON"
}
if (!$build_tools) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_BUILD_TOOLS=OFF"
}
if ($build_quantools) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_BUILD_QUANTOOLS=ON"
}
if ($build_evaluation) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_EVALUATION=ON"
}
if ($build_converter) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_BUILD_CONVERTER=ON"
    if ($dynamic_link) {
        $CMAKE_ARGS = "$CMAKE_ARGS -Dprotobuf_BUILD_SHARED_LIBS=ON"
    } else {
        $CMAKE_ARGS = "$CMAKE_ARGS -Dprotobuf_BUILD_SHARED_LIBS=OFF"
    }
}
if ($build_benchmark) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_BUILD_BENCHMARK=ON"
}
if ($build_test) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_BUILD_TEST=ON"
}
if ($build_demo) {
    $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_BUILD_DEMO=ON"
}
if ($backends -ne $null) {
    Foreach ($backend in $backends.Split(",")) {
        if ($backend -eq "opencl") {
            $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_OPENCL=ON"
        } elseif ($backend -eq "vulkan") {
            $CMAKE_ARGS = "$CMAKE_ARGS -DMNN_VULKAN=ON"
        }
    }
}

Remove-Item build -Recurse -ErrorAction Ignore
mkdir build
pushd build

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
    popd
    exit 1
}

Remove-Item CMakeCache.txt -ErrorAction Ignore
Build "cmake -G Ninja $CMAKE_ARGS  .."

$PRODUCTS = $(Get-ChildItem -Path . -Include "*.exe" -Name)
if ($dynamic_link) {
    $PRODUCTS = "$PRODUCTS MNN.dll"
    if ($build_converter) {
        $PRODUCTS = "$PRODUCTS ./3rd_party/protobuf/cmake/libprotobuf.dll"
    }
}

Foreach ($PRODUCT in $PRODUCTS.Trim().Split()) {
    Invoke-Expression "cp $PRODUCT $TOOLS_PATH"
}

popd