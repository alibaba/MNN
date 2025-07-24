# MNN TTS

This directory contains a minimal cross-platform C++ template. The code shows how
shared sources can be combined with platform specific implementations.

The library builds on desktop platforms, Android (JNI) and iOS. Android builds
can be packaged into an AAR and iOS builds can be packaged as a framework.

## Building

```
mkdir build && cd build
cmake ..
make
ctest
```

The generated library can be linked by applications on each platform.

### Build Android AAR

```
cd android
./gradlew assembleRelease
```

The resulting AAR can be found under `android/build/outputs/aar`.
