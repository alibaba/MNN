$erroractionpreference = "stop"

Set-Variable -Name WINDOWS_PACKAGE_NAME -Value "MNN_Windows_lib_and_tools"

#clear and create package directory
Set-Variable -Name WINDOWS_PACKAGE_PATH -Value "$(pwd)\$WINDOWS_PACKAGE_NAME"
Remove-Item $WINDOWS_PACKAGE_PATH -Recurse -ErrorAction Ignore
mkdir $WINDOWS_PACKAGE_PATH
cd $WINDOWS_PACKAGE_PATH
mkdir Dynamic_Library
mkdir Static_Library
mkdir tools
cd ..

.\schema\generate.ps1
Remove-Item build -Recurse -ErrorAction Ignore
mkdir build
cd build
# tools without dependency, static library without sep_build
cmake -G "Ninja" -DMNN_SEP_BUILD=OFF -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_CONVERTER=ON -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_DEMO=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON ..
ninja
pushd $WINDOWS_PACKAGE_PATH
cp ..\build\*.exe tools
cp ..\build\*.pdb tools
cp ..\build\MNN.lib Static_Library
popd

#dynamic library without sep_build
rm .\CMakeCache.txt
cmake -G "Ninja" -DMNN_SEP_BUILD=OFF ..
ninja
cd ..
cp build\MNN.lib $WINDOWS_PACKAGE_PATH\Dynamic_Library
cp build\MNN.dll $WINDOWS_PACKAGE_PATH\Dynamic_Library
cp build\MNN.pdb $WINDOWS_PACKAGE_PATH\Dynamic_Library