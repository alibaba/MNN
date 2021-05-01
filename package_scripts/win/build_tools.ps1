Param(
    [Parameter(Mandatory=$true)][String]$path,
    [String]$backends
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
Remove-Item $path -Recurse -ErrorAction Ignore
mkdir -p $path
$TOOLS_PATH = $(Resolve-Path $path).Path

powershell ./schema/generate.ps1

$CMAKE_ARGS = "-DMNN_SEP_BUILD=OFF -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_TOOLS=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TEST=ON"
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
Invoke-Expression "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON -DMNN_BUILD_SHARED_LIBS=OFF .."
Retry "ninja" 2
cp MNNV2Basic.out.exe $TOOLS_PATH
cp MNNConvert.exe $TOOLS_PATH
cp testModel.out.exe $TOOLS_PATH
cp run_test.out.exe $TOOLS_PATH
cp quantized.out.exe $TOOLS_PATH
cp train.out.exe $TOOLS_PATH
cp benchmark.out.exe $TOOLS_PATH
cp benchmarkExprModels.out.exe $TOOLS_PATH
cp backendTest.out.exe $TOOLS_PATH

popd