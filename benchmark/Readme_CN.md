基于表达式构建模型并进行benchmark：
cd /path/to/MNN
mkdir build && cd build
cmake -DMNN_BUILD_BENCHMARK=true ..
make -j8

运行以下命令查看help：
./benchmarkExprModels.out help

示例：
./benchmarkExprModels.out MobileNetV1_100_1.0_224 10 0 4
./benchmarkExprModels.out MobileNetV2_100 10 0 4
./benchmarkExprModels.out ResNet_100_18 10 0 4
./benchmarkExprModels.out GoogLeNet_100 10 0 4
./benchmarkExprModels.out SqueezeNet_100 10 0 4
./benchmarkExprModels.out ShuffleNet_100_4 10 0 4

相应模型的paper链接附在头文件里，如benchmark/exprModels/MobileNetExpr.hpp

