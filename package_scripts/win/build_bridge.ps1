# MNNPyBridge
#   |-- include
#   |-- wrapper
#   |-- test (Release + Dynamic + MD)
#        |-- x64
#        |-- x86
#   |-- lib
#        |-- x64
#        |    |-- (Debug/Release x Dynamic/Static x MD/MT)
#        |
#        |-- x86
#             |-- (Debug/Release x Dynamic/Static x MD/MT)

Param(
    [Parameter(Mandatory=$true)][String]$version,
    [Parameter(Mandatory=$true)][String]$pyc_env,
    [Parameter(Mandatory=$true)][String]$mnn_path,
    [Parameter(Mandatory=$true)][String]$python_path,
    [Parameter(Mandatory=$false)][String]$numpy_path,
    [Parameter(Mandatory=$true)][String]$path,
    [Switch]$train_api,
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
    popd
    exit 1
}

$erroractionpreference = "stop"
mkdir -p $path -ErrorAction Ignore
$PACKAGE_PATH = $(Resolve-Path $path).Path
$arch = $(If($x86) {"x86"} Else {"x64"})
$PACKAGE_LIB_PATH = "$PACKAGE_PATH/lib/$arch"
$TEST_TOOL_PATH = "$PACKAGE_PATH/test/$arch"

#clear and create package directory
pushd $PACKAGE_PATH
Remove-Item -Path include, wrapper -Recurse -ErrorAction Ignore
mkdir -p include, wrapper
popd
Remove-Item -Path $PACKAGE_LIB_PATH, $TEST_TOOL_PATH -Recurse -ErrorAction Ignore
mkdir -p $PACKAGE_LIB_PATH, $TEST_TOOL_PATH
pushd $PACKAGE_LIB_PATH
mkdir -p Debug\Dynamic\MD, Debug\Dynamic\MT, Debug\Static\MD, Debug\Static\MT, Release\Dynamic\MD, Release\Dynamic\MT, Release\Static\MD, Release\Static\MT
popd

# assume $PACKAGE_PATH exist
cp pymnn\src\MNNPyBridge.h $PACKAGE_PATH\include
Remove-Item pymnn_pyc_tmp -Recurse -ErrorAction Ignore
mkdir pymnn_pyc_tmp
cp -r pymnn\pip_package\MNN pymnn_pyc_tmp
pushd pymnn_pyc_tmp
Remove-Item MNN -Include __pycache__ -Recurse
pushd MNN
function Remove([String]$module) {
  rm -r -force $module
  (Get-Content __init__.py).replace("from . import $module", "") | Set-Content __init__.py
}
Remove "tools"
if (!$train_api) {
  Remove "data"
  Remove "optim"
}

popd
popd


$mnn_path = $(Resolve-Path $mnn_path).Path
$python_path = $(Resolve-Path $python_path).Path

$CMAKE_ARGS = "-DPYMNN_USE_ALINNPYTHON=ON -DPYMNN_RUNTIME_CHECK_VM=ON -DPYMNN_EXPR_API=ON -DPYMNN_BUILD_TEST=OFF -DPYMNN_IMGCODECS=ON  -DPYMNN_IMGPROC_DRAW=ON  -DPYMNN_IMGPROC_STRUCTURAL=ON  -DPYMNN_IMGPROC_MISCELLANEOUS=ON -DPYMNN_IMGPROC_COLOR=ON -DPYMNN_IMGPROC_GEOMETRIC=ON -DPYMNN_IMGPROC_FILTER=ON"
if ($train_api) {
  $CMAKE_ARGS = "$CMAKE_ARGS -DPYMNN_TRAIN_API=ON"
}

if ($numpy_path) {
  $CMAKE_ARGS = "$CMAKE_ARGS -DPYMNN_NUMPY_USABLE=ON -Dnumpy_path=$numpy_path"
}


$CMAKE_ARGS = "$CMAKE_ARGS -Dmnn_path=$mnn_path -Dpython_path=$python_path"

Remove-Item pymnn_build -Recurse -ErrorAction Ignore
mkdir pymnn_build
pushd pymnn_build

function exist([String]$build_type, [String]$lib_type, [String]$crt_type) {
  function _exist([String]$lib) {
    $lib_dir = "$lib/lib/$arch/$build_type/$lib_type/$crt_type"
    return $((Test-Path -Path $lib_dir) -and ((Get-ChildItem -Path "$lib_dir/*" -Include "*.lib").Count -ne 0))
  }
  return $((_exist $mnn_path) -and (_exist $python_path))
}

function log([String]$msg) {
    echo "================================"
    echo "Build MNNPyBridge $msg"
    echo "================================"
}

##### Debug/Dynamic/MT ####
if (exist Debug Dynamic MT) {
  log "Debug/Dynamic/MT"
  Remove-Item CMakeCache.txt -ErrorAction Ignore
  Build "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_WIN_RUNTIME_MT=ON ../pymnn"
  cp mnnpybridge.lib,mnnpybridge.dll,mnnpybridge.pdb $PACKAGE_LIB_PATH\Debug\MT
  rm mnnpybridge.*
}

##### Debug/Dynamic/MD ####
if (exist Debug Dynamic MD) {
  log "Debug/Dynamic/MD"
  Remove-Item CMakeCache.txt -ErrorAction Ignore
  Build "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug ../pymnn"
  cp mnnpybridge.lib,mnnpybridge.dll,mnnpybridge.pdb $PACKAGE_LIB_PATH\Debug\MD
  rm mnnpybridge.*
}

##### Debug/Static/MT ####
if (exist Debug Static MT) {
  log "Debug/Static/MT"
  Remove-Item CMakeCache.txt -ErrorAction Ignore
  Build "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_WIN_RUNTIME_MT=ON -DMNN_BUILD_SHARED_LIBS=OFF ../pymnn"
  cp mnnpybridge.lib $PACKAGE_LIB_PATH\Debug\Static\MT
  rm mnnpybridge.*
}

##### Debug/Static/MD ####
if (exist Debug Static MD) {
  log "Debug/Static/MD"
  Remove-Item CMakeCache.txt -ErrorAction Ignore
  Build "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DMNN_BUILD_SHARED_LIBS=OFF ../pymnn"
  cp mnnpybridge.lib $PACKAGE_LIB_PATH\Debug\Static\MD
  rm mnnpybridge.*
}

##### Release/Dynamic/MT ####
if (exist Release Dynamic MT) {
  log "Release + MT"
  Remove-Item CMakeCache.txt -ErrorAction Ignore
  Build "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON ../pymnn"
  cp mnnpybridge.lib,mnnpybridge.dll,mnnpybridge.pdb $PACKAGE_LIB_PATH\Release\Dynamic\MT
  rm mnnpybridge.*
}

##### Release/Dynamic/MD ####
if (exist Release Dynamic MD) {
  log "Release + MD"
  Remove-Item CMakeCache.txt -ErrorAction Ignore
  Build "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release ../pymnn"
  cp mnnpybridge.lib,mnnpybridge.dll,mnnpybridge.pdb $PACKAGE_LIB_PATH\Release\Dynamic\MD
  #cp mnnpybridge_test.exe $TEST_TOOL_PATH
  #cp $mnn_path/lib/$arch/Release/MD/MNN.dll $TEST_TOOL_PATH
  #cp $python_path/lib/$arch/Release/MD/python.dll $TEST_TOOL_PATH
  #cp $numpy_path/lib/$arch/Release/MD/numpy_python.dll $TEST_TOOL_PATH
  rm mnnpybridge.*
}

##### Release/Static/MT ####
if (exist Release Static MT) {
  log "Release/Static/MT"
  Remove-Item CMakeCache.txt -ErrorAction Ignore
  Build "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_WIN_RUNTIME_MT=ON -DMNN_BUILD_SHARED_LIBS=OFF ../pymnn"
  cp mnnpybridge.lib $PACKAGE_LIB_PATH\Release\Static\MT
  rm mnnpybridge.*
}

##### Release/Static/MD ####
if (exist Release Static MD) {
  log "Release/Static/MD"
  Remove-Item CMakeCache.txt -ErrorAction Ignore
  Build "cmake -G Ninja $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF ../pymnn"
  cp mnnpybridge.lib $PACKAGE_LIB_PATH\Release\Static\MD
  rm mnnpybridge.*
}

popd