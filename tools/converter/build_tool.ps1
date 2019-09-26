$erroractionpreference = "stop"

if (Test-Path "build" -PathType Container) {
  rm -r -force build
}
.\generate_schema.ps1

mkdir build
cd build

cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF ..
ninja
