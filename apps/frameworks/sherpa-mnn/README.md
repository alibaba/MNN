# sherpa-mnn

本工程基于 sherpa-onnx 改造而得，将 onnxruntime 的调用全部替换为 MNN

## MNN 环境和模型准备

### MNN 编译

下载 MNN : https://github.com/alibaba/MNN/

在编译 MNN 时额外加上 `-DMNN_SEP_BUILD=OFF` 和 `-DCMAKE_INSTALL_PREFIX=.` ，:

```
mkdir build
cd build
cmake .. -DMNN_LOW_MEMORY=ON -DMNN_SEP_BUILD=OFF -DCMAKE_INSTALL_PREFIX=. -DMNN_BUILD_CONVERTER=ON
make -j4
make install
```

### 模型转换
在 编译好 MNNConvert 的目录下（上文的build目录），按如下命令逐个把下载好的 onnx FP32 模型转换成 mnn ，建议转换时量化一下，可以降低模型大小，并在MNN库开启`MNN_LOW_MEMORY`编译的情况下降低运行内存并提升运行性能，不要直接转换 int8 的 onnx 模型。
```
mkdir sherpa-mnn-models
./MNNConvert -f ONNX --modelFile  sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx --MNNModel sherpa-mnn-models/encode.mnn --weightQuantBits=8 --weightQuantBlock=64
./MNNConvert -f ONNX --modelFile  sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx --MNNModel sherpa-mnn-models/decode.mnn --weightQuantBits=8 --weightQuantBlock=64
./MNNConvert -f ONNX --modelFile  sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx --MNNModel sherpa-mnn-models/joiner.mnn --weightQuantBits=8 --weightQuantBlock=64
```


## 本地编译和运行测试

### 编译
回到 sherpa-mnn 根目录
执行如下操作, `MNN_LIB_DIR`后面的内容按自己的编译目录修改

```
mkdir build
cmake .. -DMNN_LIB_DIR=/Users/xtjiang/alicnn/AliNNPrivate/build
make -j16
```

### 测试
回到 sherpa-mnn 根目录，以sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 这个模型为例

```
./build/bin/sherpa-mnn  --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt   --encoder=./sherpa-mnn-models/encode.mnn   --decoder=./sherpa-mnn-models/decode.mnn   --joiner=./sherpa-mnn-models/joiner.mnn   ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/1.wav
```

正常的话会打印如下信息
```
Number of threads: 1, Elapsed seconds: 0.27, Audio duration (s): 5.1, Real time factor (RTF) = 0.27/5.1 = 0.053
这是第一种第二种叫与 ALWAYS ALWAYS什么意思
{ "text": "这是第一种第二种叫与 ALWAYS ALWAYS什么意思", "tokens": ["这", "是", "第", "一", "种", "第", "二", "种", "叫", "与", " ALWAYS", " ALWAYS", "什", "么", "意", "思"], "timestamps": [0.96, 1.04, 1.28, 1.40, 1.48, 1.72, 1.84, 2.04, 2.44, 3.64, 3.84, 4.36, 4.72, 4.76, 4.92, 5.04], "ys_probs": [-0.884769, -0.858386, -1.106216, -0.626572, -1.101773, -0.359962, -0.745972, -0.267809, -0.826859, -1.076653, -0.683002, -0.869667, -0.593140, -0.469688, -0.256882, -0.442532], "lm_probs": [], "context_scores": [], "segment": 0, "words": [], "start_time": 0.00, "is_final": false}
```

## 编译 Android
### MNN Android 编译
进入 MNN 目录后操作
```
cd project/android
mkdir build_64
../build_64.sh -DMNN_LOW_MEMORY=ON -DMNN_SEP_BUILD_OFF -DCMAKE_INSTALL_PREFIX=.
make install
```

### sherpa-mnn Android 编译
修改 build-android-arm64-v8a.sh 脚本
把 `MNN_LIB_DIR`后面的内容修改为上面的编译目录

然后执行 build-android-arm64-v8a.sh

如果编译出来的 so 体积较大，可以用 android ndk 工具 strip 一下


## 编译 iOS
修改 build-ios.sh 脚本
把 `MNN_LIB_DIR`后面的内容修改为 MNN 根目录（保证能找到 MNN 头文件即可）

运行 build-ios.sh 脚本

```
export MNN_LIB_DIR=/path/to/MNN
sh build-ios.sh
```

编译出 build-ios/sherpa-mnn.xcframework

## 编译 MacOs framework
类似 iOS 编译过程，修改 build-swift-macos.sh
把 `MNN_LIB_DIR`后面的内容修改为 MNN 根目录（保证能找到 MNN 头文件即可）
运行 build-swift-macos.sh
编译出 build-swift-macos/sherpa-mnn.xcframework/
