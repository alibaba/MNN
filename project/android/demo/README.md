# MNN Android Demo

## 1. 环境准备

### 开发工具
- `Android Studio`
- `NDK`

### 模型下载与转换：
首先编译(如果已编译可以跳过)`MNNConvert`，操作如下：
```
cd MNN
mkdir build && cd build
cmake -DMNN_BUILD_CONVERTER=ON ..
make -j8
```

然后下载模型，可以直接执行 sh ../tools/script/get_model.sh ，也可以按如下步骤自行下载与转换：
#### MobileNet_v2
```
wget https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet_v2.caffemodel
wget https://github.com/shicai/MobileNet-Caffe/blob/master/mobilenet_v2_deploy.prototxt
./MNNConvert -f CAFFE --modelFile mobilenet_v2.caffemodel --prototxt mobilenet_v2_deploy.prototxt --MNNModel mobilenet_v2.caffe.mnn
mv mobilenet_v2.caffe.mnn ../resource/model/MobileNet/v2/
```

#### SqueezeNet_v1.1
```
wget https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
wget https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.1/deploy.prototxt
./MNNConvert -f CAFFE --modelFile squeezenet_v1.1.caffemodel --prototxt deploy.prototxt --MNNModel squeezenet_v1.1.caffe.mnn
mv squeezenet_v1.1.caffe.mnn ../resource/model/SqueezeNet/v1.1/
```
#### DeepLab_v3
```
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite
./MNNConvert -f TFLITE --modelFile deeplabv3_257_mv_gpu.tflite --MNNModel Portrait.tflite.mnn
mv Portrait.tflite.mnn ../resource/model/Portrait/
```

## 2. 编译运行

使用`Android Studio`打开`demo`目录，在`local.properties`中指定`sdk.dir`与`ndk.dir`，即可编译执行。
