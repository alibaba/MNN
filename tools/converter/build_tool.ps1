$erroractionpreference = "stop"

if (Test-Path "build" -PathType Container) {
  rm -r -force build
}
.\generate_schema.ps1

mkdir build
cd build

cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
nmake
