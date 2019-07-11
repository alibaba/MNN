[TOC]

# Build this Demo on Linux or Mac

Use [Top CMakeLists.txt](../../CMakeLists.txt) to construct demo like this:

```bash
cd path/to/MNN
mkdir build && cd build
cmake -DMNN_BUILD_DEMO=ON ..
make -j8
```

# Build this Demo on Windows

Use [Top CMakeLists.txt](../../CMakeLists.txt) to construct demo like this:
```powershell
cd path/to/MNN
mkdir build
cd build
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_DEMO=ON ..
nmake
```

# MultiPose

1. Download [pose model](https://github.com/czy2014hust/posenet-python/raw/master/models/model-mobilenet_v1_075.pb)
2. [Convert](../../tools/converter/README.md) tensorflow model to MNN model
3. run multipose like this:
```bash
./multiPose.out model.mnn input.jpg pose.png
```

# Segment
Using deeplabv3 segment model downloaded from:
https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite
```bash
./segment.out model.mnn input.jpg result.png
```
