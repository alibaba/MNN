[TOC]

# Build this Demo on Linux or Mac

1. Install freeimage lib
```bash
# Linux
sudo apt-get install libfreeimage
# Mac
brew install freeimage
```

2. cmake MNN and this Demo

Use [Top CMakeLists.txt](../../CMakeLists.txt) to construct demo like this:

```bash
cd path/to/MNN
mkdir build
cmake -DMNN_FREEIMAGE_VALID=ON ..
make -j8
```

# MultiPose

1. Download [pose model](https://github.com/czy2014hust/posenet-python/raw/master/models/model-mobilenet_v1_075.pb)
2. [Convert](../../tools/converter/README.md) tensorflow model to MNN model
3. run multipose like this:
```bash
./multiPose.out model.mnn input.jpg pose.jpg
```