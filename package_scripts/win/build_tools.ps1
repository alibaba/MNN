Param(
    [Parameter(Mandatory=$true)][String]$path,
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
Remove-Item $path -Recurse -ErrorAction Ignore
mkdir -p $path
$TOOLS_PATH = $(Resolve-Path $path).Path

powershell ./schema/generate.ps1

$CMAKE_ARGS = "-DMNN_SEP_BUILD=OFF -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON -DMNN_BUILD_SHARED_LIBS=OFF"
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

Remove-Item CMakeCache.txt -ErrorAction Ignore
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS  .."
Retry "ninja" 2

$PRODUCTS = ""
if ($build_train) {
    $PRODUCTS = "$PRODUCTS transformer.out.exe train.out.exe rawDataTransform.out.exe dataTransformer.out.exe runTrainDemo.out.exe"
}
if ($build_tools) {
    $PRODUCTS = "$PRODUCTS MNNV2Basic.out.exe mobilenetTest.out.exe backendTest.out.exe testModel.out.exe testModelWithDescrisbe.out.exe getPerformance.out.exe checkInvalidValue.out.exe timeProfile.out.exe"
}
if ($build_quantools) {
    $PRODUCTS = "$PRODUCTS quantized.out.exe quantized_model_optimize.out.exe"
}
if ($build_evaluation) {
    $PRODUCTS = "$PRODUCTS classficationTopkEval.out.exe"
}
if ($build_converter) {
    $PRODUCTS = "$PRODUCTS MNNDump2Json.exe MNNConvert.exe"
}
if ($build_benchmark) {
    $PRODUCTS = "$PRODUCTS benchmark.out.exe benchmarkExprModels.out.exe"
}
if ($build_test) {
    $PRODUCTS = "$PRODUCTS run_test.out.exe"
}
if ($build_demo) {
    $PRODUCTS = "$PRODUCTS pictureRecognition.out.exe pictureRotate.out.exe multiPose.out.exe segment.out.exe expressDemo.out.exe transformerDemo.out.exe rasterDemo.out.exe"
}

Foreach ($PRODUCT in $PRODUCTS.Split(" ")) {
    Invoke-Expression "cp $PRODUCT $TOOLS_PATH"
}

popd