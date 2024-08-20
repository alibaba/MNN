~~~
mkdir build 
cd build
cmake .. -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_TEST=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_PORTABLE_BUILD=ON -DTFMODEL_OPTIMIZE=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_LOW_MEMORY=ON -DMNN_AVX512=ON
make -j20
~~~

## 其他模型转换到MNN
### TensorFlow to MNN
```bash
./MNNConvert -f TF --modelFile XXX.pb --MNNModel XXX.mnn --bizCode biz
```
注意：`*.pb`必须是frozen model，不能使用saved_model

### TensorFlow Lite to MNN
```bash
./MNNConvert -f TFLITE --modelFile XXX.tflite --MNNModel XXX.mnn --bizCode biz
```

e.g.
~~~
cd build

./MNNConvert -f TF --modelFile ../example/model-mobilenet_v1_075.pb --MNNModel ../example/pose_model.mnn --bizCode biz --optimizeLevel 1 --optimizePrefer 2

python3 ../tools/script/testMNNFromTf.py ../example/model-mobilenet_v1_075.pb

# check supported Ops
./MNNConvert -f TF --OP
~~~