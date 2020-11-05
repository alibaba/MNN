call "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars32.bat"
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
ninja