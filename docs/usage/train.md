# 训练框架使用
[从源码编译](../compile/tools.html#id3)
## 使用表达式接口训练
### 获取可训练模型
首先我们需要获得一个可训练的模型结构，而在MNN中可以用两种方式达到这一目的。

一、将其他框架，如TensorFlow，Pytorch训练得到的模型转成MNN可训练模型，这一过程可使用 MNNConverter 工具实现。典型的应用场景为 1. 使用MNN Finetune，2. 使用MNN进行训练量化。在模型转换过程中建议使用 MNNConverter 的 --forTraining 选项，保留BatchNorm，Dropout等训练过程中会用到的算子。

二、使用MNN从零开始搭建一个模型，并使用MNN进行训练，这可以省去模型转换的步骤，并且也可以十分容易地转换为训练量化模型。在 MNN_ROOT/tools/train/source/models/ 目录中我们提供了Lenet，MobilenetV1，MobilenetV2等使用MNN训练框架搭建的模型。

#### 将其他框架模型转换为MNN可训练模型
以MobilenetV2的训练量化为例。首先我们需要到下载TensorFlow官方提供的MobilenetV2模型，然后编译 MNNConverter，并执行以下命令进行转换：
```bash
./MNNConvert --modelFile mobilenet_v2_1.0_224_frozen.pb  --MNNModel mobilenet_v2_tfpb_train.mnn --framework TF --bizCode AliNNTest --forTraining
```
注意，上述命令中使用到的 mobilenet_v2_1.0_224_frozen.pb 模型中含有 BatchNorm 算子，没有进行融合，通过在转换时使用 --forTraining 选项，我们保留了BatchNorm算子到转换出来的 mobilenet_v2_tfpb_train.mnn 模型之中。

如果你的模型中没有BN，Dropout等在转MNN推理模型时会被融合掉的算子，那么直接使用MNN推理模型也可以进行训练，不必重新进行转换。

接下来我们仿照 MNN_ROOT/tools/train/source/demo/mobilenetV2Train.cpp 中的示例，读取转换得到的模型，将其转换为MNN可训练模型。关键代码示例如下
```cpp
// mobilenetV2Train.cpp

// 读取转换得到的MNN模型
auto varMap = Variable::loadMap(argv[1]);
// 指定量化bit数
int bits = 8;
// 获取模型的输入和输出
auto inputOutputs = Variable::getInputAndOutput(varMap);
auto inputs       = Variable::mapToSequence(inputOutputs.first);
auto outputs      = Variable::mapToSequence(inputOutputs.second);

// 将转换得到的模型转换为可训练模型(将推理模型中的卷积，BatchNorm，Dropout抽取出来，转换成可训练模块)
std::shared_ptr<Module> model(PipelineModule::extract(inputs, outputs, true));
// 将可训练模型转换为训练量化模型，如果不需要进行训练量化，则可不做这一步
((PipelineModule*)model.get())->toTrainQuant(bits);
// 进入训练环节
MobilenetV2Utils::train(model, 1001, 1, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt);
```

#### 使用MNN从零开始搭建模型
以Lenet为例，我们来看一下，如何使用MNN从零搭建一个模型。MNN提供了丰富的算子可供使用，下面的例子就不详细展开。值得注意的是Pooling输出为NC4HW4格式，需要转换到 NCHW 格式才能进入全连接层进行计算。
```cpp
class MNN_PUBLIC Lenet : public Module {
public:
    Lenet();

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;

    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> ip1;
    std::shared_ptr<Module> ip2;
    std::shared_ptr<Module> dropout;
};
// 初始化
Lenet::Lenet() {
    NN::ConvOption convOption;
    convOption.kernelSize = {5, 5};
    convOption.channel    = {1, 20};
    conv1.reset(NN::Conv(convOption));
    convOption.reset();
    convOption.kernelSize = {5, 5};
    convOption.channel    = {20, 50};
    conv2.reset(NN::Conv(convOption));
    ip1.reset(NN::Linear(800, 500));
    ip2.reset(NN::Linear(500, 10));
    dropout.reset(NN::Dropout(0.5));
    // 必须要进行register的参数才会进行更新
    registerModel({conv1, conv2, ip1, ip2, dropout});
}
// 前向计算
std::vector<Express::VARP> Lenet::onForward(const std::vector<Express::VARP>& inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x      = conv1->forward(x);
    x      = _MaxPool(x, {2, 2}, {2, 2});
    x      = conv2->forward(x);
    x      = _MaxPool(x, {2, 2}, {2, 2});
    // Pooling输出为NC4HW4格式，需要转换到NCHW才能进入全连接层进行计算
    x      = _Convert(x, NCHW);
    x      = _Reshape(x, {0, -1});
    x      = ip1->forward(x);
    x      = _Relu(x);
    x      = dropout->forward(x);
    x      = ip2->forward(x);
    x      = _Softmax(x, 1);
    return {x};
}
```

### 实现数据集接口
这部分在MNN文档中[加载训练数据](train.html#id8) 部分有详细描述。

### 训练并保存模型
以MNIST模型训练为例，代码在 MNN_ROOT/tools/train/source/demo/MnistUtils.cpp

```cpp
//  MnistUtils.cpp

......

void MnistUtils::train(std::shared_ptr<Module> model, std::string root) {
    {
        // Load snapshot
        // 模型结构 + 模型参数
        auto para = Variable::load("mnist.snapshot.mnn");
        model->loadParameters(para);
    }
    // 配置训练框架参数
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    // 使用CPU，4线程
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
    // SGD求解器
    std::shared_ptr<SGD> sgd(new SGD(model));
    // SGD求解器参数设置
    sgd->setMomentum(0.9f);
    sgd->setWeightDecay(0.0005f);

    // 创建数据集和DataLoader
    auto dataset = MnistDataset::create(root, MnistDataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    const size_t batchSize  = 64;
    const size_t numWorkers = 0;
    bool shuffle            = true;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    size_t iterations = dataLoader->iterNumber();

    auto testDataset            = MnistDataset::create(root, MnistDataset::Mode::TEST);
    const size_t testBatchSize  = 20;
    const size_t testNumWorkers = 0;
    shuffle                     = false;

    auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    size_t testIterations = testDataLoader->iterNumber();

    // 开始训练
    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            // 训练阶段需设置isTraining Flag为true
            model->setIsTraining(true);
            Timer _100Time;
            int lastIndex = 0;
            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                // AUTOTIME;
                // 获得一个batch的数据，包括数据及其label
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto cast       = _Cast<float>(example.first[0]);
                example.first[0] = cast * _Const(1.0f / 255.0f);
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(10), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));
				// 前向计算
                auto predict = model->forward(example.first[0]);
                // 计算loss
                auto loss    = _CrossEntropy(predict, newTarget);
                // 调整学习率
                float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);
                
                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
                    std::cout << "epoch: " << (epoch);
                    std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate;
                    std::cout << " time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << (i - lastIndex) <<  " iter"  << std::endl;
                    std::cout.flush();
                    _100Time.reset();
                    lastIndex = i;
                }
                // 根据loss反向计算，并更新网络参数
                sgd->step(loss);
            }
        }
        // 保存模型参数，便于重新载入训练
        Variable::save(model->parameters(), "mnist.snapshot.mnn");
        {
            model->setIsTraining(false);
            auto forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
            forwardInput->setName("data");
            auto predict = model->forward(forwardInput);
            predict->setName("prob");
            // 优化网络结构【可选】
            Transformer::turnModelToInfer()->onExecute({predict});
            // 保存模型和结构，可脱离Module定义使用
            Variable::save({predict}, "temp.mnist.mnn");
        }
		
        // 测试模型
        int correct = 0;
        testDataLoader->reset();
        // 测试时，需设置标志位
        model->setIsTraining(false);
        int moveBatchSize = 0;
        for (int i = 0; i < testIterations; i++) {
            auto data       = testDataLoader->next();
            auto example    = data[0];
            moveBatchSize += example.first[0]->getInfo()->dim[0];
            if ((i + 1) % 100 == 0) {
                std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
            }
            auto cast       = _Cast<float>(example.first[0]);
            example.first[0] = cast * _Const(1.0f / 255.0f);
            auto predict    = model->forward(example.first[0]);
            predict         = _ArgMax(predict, 1);
            auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.second[0]))).sum({});
            correct += accu->readMap<int32_t>()[0];
        }
        // 计算准确率
        auto accu = (float)correct / (float)testDataLoader->size();
        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
        exe->dumpProfile();
    }
}

```

### 保存和恢复模型
一、只保存模型参数，不保存模型结构，需要对应的模型结构去加载这些参数
保存：
```cpp
Variable::save(model->parameters(), "mnist.snapshot.mnn");
```
恢复：
```cpp
// 模型结构 + 模型参数
auto para = Variable::load("mnist.snapshot.mnn");
model->loadParameters(para);
```

二、同时保存模型结构和参数，便于推理
保存：
```cpp
model->setIsTraining(false);
auto forwardInput = _Input({1, 1, 28, 28}, NC4HW4);
forwardInput->setName("data");
auto predict = model->forward(forwardInput);
predict->setName("prob");

// 保存输出节点，会连同结构参数一并存储下来
Variable::save({predict}, "temp.mnist.mnn");
```
恢复（进行推理）：
```cpp
auto varMap = Variable::loadMap("temp.mnist.mnn");

//输入节点名与保存时设定的名字一致，为 data，维度大小与保存时设定的大小一致，为 [1, 1, 28, 28]
float* inputPtr = varMap["data"]->writeMap<float>();
//填充 inputPtr

//输出节点名与保存时设定的名字一致，为 prob
float* outputPtr = varMap["prob"]->readMap<float>();
// 使用 outputPtr 的数据

```

## 加载训练数据
该模块用于读取保存在硬盘上的数据，将其包装并输出为MNN训练可用的数据类型。该模块源码位于MNN_root/tools/train/source/data/目录下。若要使用，请包含DataLoader.hpp头文件即可，该模块中其他组件会全部导入，用于构建DataLoader。

### 相关demo
1、MNN_root/tools/train/source/demo/dataLoaderDemo.cpp
    使用MNIST数据集构建DataLoader，并进行输出显示。
2、MNN_root/tools/train/source/demo/dataLoaderTest.cpp
    使用MNIST数据集构建DataLoader，并测试DataLoader中一些组件。
3、MNN_root/tools/train/source/demo/ImageDatasetDemo.cpp
    读取硬盘上保存的图片数据，并显示出来。显示需要用到OpenCV，并在编译时打开`MNN_USE_OPENCV`宏。

### 自定义Dataset
可参考MNN_root/tools/train/source/datasets/中预置数据集的写法，继承Dataset类，实现两个抽象函数即可，例如：
```cpp
//  MnistDataset.cpp

// 返回MNIST数据集中一张图片数据，及其对应的label
Example MnistDataset::get(size_t index) {
    auto data  = _Input({1, kImageRows, kImageColumns}, NCHW, halide_type_of<uint8_t>());
    auto label = _Input({}, NCHW, halide_type_of<uint8_t>());

    auto dataPtr = mImagePtr + index * kImageRows * kImageColumns;
    ::memcpy(data->writeMap<uint8_t>(), dataPtr, kImageRows * kImageColumns);

    auto labelPtr = mLabelsPtr + index;
    ::memcpy(label->writeMap<uint8_t>(), labelPtr, 1);

    auto returnIndex = _Const(index);
    // return the index for test
    return {{data, returnIndex}, {label}};
}
// 返回数据集大小，对于MNIST训练集是60000，测试集是10000
size_t MnistDataset::size() {
    return mImages->getInfo()->dim[0];
}
```

### DataLoader使用示例
使用流程：自定义Dataset，构造DataLoader，读取数据，DataLoader->reset();
```cpp
//
//  ImageDatasetDemo.cpp
//  MNN
//
//  Created by MNN on 2019/11/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include "ImageDataset.hpp"
#include "RandomSampler.hpp"
#include "Sampler.hpp"
#include "Transform.hpp"
#include "TransformDataset.hpp"

#ifdef MNN_USE_OPENCV
#include <opencv2/opencv.hpp> // use opencv to show pictures
using namespace cv;
#endif

using namespace std;

/*
 * this is an demo for how to use the ImageDataset and DataLoader
 */

class ImageDatasetDemo : public DemoUnit {
public:
    // this function is an example to use the lambda transform
    // here we use lambda transform to normalize data from 0~255 to 0~1
    static Example func(Example example) {
        // // an easier way to do this
        auto cast       = _Cast(example.first[0], halide_type_of<float>());
        example.first[0] = _Multiply(cast, _Const(1.0f / 255.0f));
        return example;
    }

    virtual int run(int argc, const char* argv[]) override {
        if (argc != 3) {
            cout << "usage: ./runTrainDemo.out ImageDatasetDemo path/to/images/ path/to/image/txt\n" << endl;

            // ImageDataset的数据格式，采用的是ImageNet数据集的格式，你也可以写一个自己的数据集，自定义格式
            cout << "the ImageDataset read stored images as input data.\n"
                    "use 'pathToImages' and a txt file to construct a ImageDataset.\n"
                    "the txt file should use format as below:\n"
                    "     image1.jpg label1,label2,...\n"
                    "     image2.jpg label3,label4,...\n"
                    "     ...\n"
                    "the ImageDataset would read images from:\n"
                    "     pathToImages/image1.jpg\n"
                    "     pathToImages/image2.jpg\n"
                    "     ...\n"
                 << endl;

            return 0;
        }

        std::string pathToImages   = argv[1];
        std::string pathToImageTxt = argv[2];

        // ImageDataset可配置数据预处理
        auto converImagesToFormat  = ImageDataset::DestImageFormat::RGB;
        int resizeHeight           = 224;
        int resizeWidth            = 224;
        std::vector<float> scales = {1/255.0, 1/255.0, 1/255.0};
        auto config                = ImageDataset::ImageConfig(converImagesToFormat, resizeHeight, resizeWidth, scales);
        bool readAllImagesToMemory = false;

        // 构建ImageDataset
        auto dataset = std::make_shared<ImageDataset>(pathToImages, pathToImageTxt, config, readAllImagesToMemory);

        const int batchSize  = 1;
        const int numWorkers = 1;

        // 构建DataLoader，这里会将一个batch数据stack为一个VARP(Tensor)
        auto dataLoader = std::shared_ptr<DataLoader>(DataLoader::makeDataLoader(dataset, batchSize, true, false, numWorkers));

        const size_t iterations = dataset->size() / batchSize;

        for (int i = 0; i < iterations; i++) {
            // 读取数据
            auto trainData = dataLoader->next();

            auto data  = trainData[0].first[0]->readMap<float_t>();
            auto label = trainData[0].second[0]->readMap<int32_t>();

            cout << "index: " << i << " label: " << int(label[0]) << endl;

#ifdef MNN_USE_OPENCV
            // only show the first picture in the batch
            Mat image = Mat(resizeHeight, resizeWidth, CV_32FC(3), (void*)data);
            imshow("image", image);

            waitKey(-1);
#endif
        }
        
        // 每完整过一次数据集必须重置DataLoader
        // this will reset the sampler's internal state
        dataLoader->reset();
        return 0;
    }
};

DemoUnitSetRegister(ImageDatasetDemo, "ImageDatasetDemo");
```

### 相关类和概念
#### VARP
MNN动态图中的变量，类似于pytorch中的Tensor

#### Example
DataLoader输出数据的最小单位

```cpp
/**
 First: data: a vector of input tensors (for single input dataset is only one)
 Second: target: a vector of output tensors (for single output dataset is only one)
 */
typedef std::pair<std::vector<VARP>, std::vector<VARP>> Example;
```
可以看到一个Example是一个数据对，其first部分是输入，second部分是target。由于网络有可能有多个输入和多个target，所以first和second都是vector结构。

#### RandomSampler : public Sampler
随机采样序列生成器，例如图片数据集中有1000张图片，则生成采样序列0~999，根据配置指定是否进行shuffle
```cpp
public:
	// size: 采样序列长度
	// shuffle: 是否生成随机采样序列
    explicit RandomSampler(size_t size, bool shuffle = true);
	// 重置采样器内部状态
    void reset(size_t size) override;
	// 采样器长度
    size_t size() override;
	// 返回内部生成的采样序列
    const std::vector<size_t> indices();
	// 返回已经使用的采样序列数量
    size_t index();
	// 获取下一个长度为batchSize的采样序列
    std::vector<size_t> next(size_t batchSize) override;

private:
    std::vector<size_t> mIndices;
    size_t mIndex = 0;
    bool mShuffle;
```

#### Dataset
数据集抽象基类，用户自定义数据集需继承此基类，并实现抽象函数，可参考MNN_root/tools/train/source/datasets/中预置数据集的写法
```cpp
// 返回数据集的大小，例如1000张图片的数据集，其大小为1000
virtual size_t size() = 0;
// 返回数据集中指定index的数据，如给定123，返回第123张图片数据
virtual Example get(size_t index) = 0;
// 返回数据集中指定index的一批数据，为一个batch
std::vector<Example> getBatch(std::vector<size_t> indices);
```

#### Transform
抽象基类，对一个batch中的每一个数据进行某个变换，可以是一些预处理等

#### BatchTransform
抽象基类，对一个batch的数据进行某个变换，可以是一些预处理等

#### StackTransform : public BatchTransform
将一个Dataset输出的vector<Example>表示的一个batch数据合成一个VARP，即
Stack( (c, h, w), (c, h, w), (c, h, w)... ) --> (n, c, h, w)

#### LambdaTransform : public Transform
对Dataset输出的每一个Example进行单独处理，例如中心化，归一化等

#### TransformDataset : public Dataset
对Dataset进行某种Transform，仍是一个Dataset，用于输出数据

#### DataLoaderConfig
对DataLoader进行配置，可配置项为：
> batchSize: 指定batch大小
> numWorkers: 多线程预读取的线程数

#### DataLoader
根据采样器生成的采样序列，到对应的Dataset中取得对应的数据并输出
```cpp
// 构造函数
DataLoader(std::shared_ptr<BatchDataset> dataset, std::shared_ptr<Sampler> sampler,
               std::shared_ptr<DataLoaderConfig> config);
// 构造DataLoader，无Transform
static DataLoader* makeDataLoader(std::shared_ptr<BatchDataset> dataset,
                                  const int batchSize,
                                  const bool stack = true, // 是否将一个batch数据叠加为一个VARP(Tensor)
                                  const bool shuffle = true,
                                  const int numWorkers = 0);
// 构造DataLoader，有Transform，Transform可多个叠加
static DataLoader* makeDataLoader(std::shared_ptr<BatchDataset> dataset,
                                  std::vector<std::shared_ptr<BatchTransform>> transforms,
                                  const int batchSize,
                                  const bool shuffle = true,
                                  const int numWorkers = 0);
// 指定batch size后，迭代多少次用完全部数据，最后一个batch不足batchsize也会输出
size_t iterNumber() const;
// 数据集大小
size_t size() const;
// 输出一个batch的数据
std::vector<Example> next();
// 清空内部数据队列，重置内部采样器
void clean();
// clean()，并重新预读取，Dataset每次数据全部输出完毕，必须reset
void reset();
```

## 优化器使用
### SGD with momentum
使用示例
```cpp
// 新建SGD优化器
std::shared_ptr<SGD> solver(new SGD);
// 设置模型中需要优化的参数
solver->append(model->parameters());
// 设置momentum和weight decay
solver->setMomentum(0.9f);
solver->setWeightDecay(0.0005f);
// 设置正则化方法，默认L2
solver->setRegularizationMethod(RegularizationMethod::L2);
// 设置学习率
solver->setLearningRate(0.001);

// 根据loss计算梯度，并更新参数
solver->step(loss);
```

### ADAM
使用示例
```cpp
// 新建ADAM优化器
std::shared_ptr<SGD> solver(new ADAM);
// 设置模型中需要优化的参数
solver->append(model->parameters());
// 设置ADAM的两个momentum，设置weight decay
solver->setMomentum(0.9f);
solver->setMomentum2(0.99f);
solver->setWeightDecay(0.0005f);
// 设置正则化方法，默认L2
solver->setRegularizationMethod(RegularizationMethod::L2);
// 设置学习率
solver->setLearningRate(0.001);

// 根据loss计算梯度，并更新参数
solver->step(loss);
```

### Loss
目前支持的Loss，也可自行设计
```cpp
VARP _CrossEntropy(Express::VARP predicts, Express::VARP oneHotTargets);

VARP _KLDivergence(Express::VARP predicts, Express::VARP oneHotTargets);

VARP _MSE(Express::VARP predicts, Express::VARP oneHotTargets);

VARP _MAE(Express::VARP predicts, Express::VARP oneHotTargets);

VARP _Hinge(Express::VARP predicts, Express::VARP oneHotTargets);

VARP _DistillLoss(Express::VARP studentLogits, Express::VARP teacherLogits, Express::VARP oneHotTargets,
                                                                const float temperature, const float alpha);
```

## 训练量化
### 什么是训练量化
与离线量化不同，训练量化需要在训练中模拟量化操作的影响，并通过训练使得模型学习并适应量化操作所带来的误差，从而提高量化的精度。因此训练量化也称为Quantization-aware Training（QAT），意指训练中已经意识到此模型将会转换成量化模型。

### 如何在MNN中使用训练量化
已经通过其他训练框架如TensorFlow、Pytorch等训练得到一个float模型，此时可以通过先将此float模型通过MNNConverter转换为MNN统一的模型格式，然后使用MNN提供的离线量化工具直接量化得到一个全int8推理模型。如果此模型的精度不满足要求，则可以通过训练量化来提高量化模型的精度。

使用步骤：
1. 首先通过其他训练框架训练得到原始float模型；
2. 编译MNNConverter模型转换工具；
3. 使用MNNConverter将float模型转成MNN统一格式模型，因为要进行再训练，建议保留BN，Dropout等训练过程中会使用到的算子，这可以通过MNNConverter的 --forTraining 选项实现；
4. 参考MNN_ROOT/tools/train/source/demo/mobilenetV2Train.cpp 中的 MobilenetV2TrainQuant demo来实现训练量化的功能，下面以MobilenetV2的训练量化为例，来看一下如何读取并将模型转换成训练量化模型
5. 观察准确率变化，代码保存下来的模型即为量化推理模型
```cpp
//  mobilenetV2Train.cpp

// 读取转换得到的MNN float模型
auto varMap = Variable::loadMap(argv[1]);
if (varMap.empty()) {
    MNN_ERROR("Can not load model %s\n", argv[1]);
    return 0;
}
// 指定量化比特数
int bits = 8;
if (argc > 6) {
    std::istringstream is(argv[6]);
    is >> bits;
}
if (1 > bits || bits > 8) {
    MNN_ERROR("bits must be 2-8, use 8 default\n");
    bits = 8;
}
// 获得模型的输入和输出
auto inputOutputs = Variable::getInputAndOutput(varMap);
auto inputs       = Variable::mapToSequence(inputOutputs.first);
auto outputs      = Variable::mapToSequence(inputOutputs.second);

// 扫描整个模型，并将inference模型转换成可训练模型，此时得到的模型是可训练的float模型
std::shared_ptr<Module> model(PipelineModule::extract(inputs, outputs, true));
// 将上面得到的模型转换成训练量化模型，此处指定量化bit数
PipelineModule::turnQuantize(model.get(), bits);
// 进行训练，观察训练结果，保存得到的模型即是量化模型
MobilenetV2Utils::train(model, 1001, 1, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt);
```
### MNN训练量化原理
MNN训练量化的基本原理如下图所示
![image.png](https://cdn.nlark.com/yuque/0/2020/png/405909/1582775538889-77cfe824-3f07-4456-a99e-b529ce888243.png#height=523&id=t2nNB&name=image.png&originHeight=1456&originWidth=1078&originalType=binary&size=590394&status=done&style=none&width=387)
以int8量化为例，首先要理解全int8推理的整个过程，全int8推理，即feature要量化为int8，weight和bias也要量化为int8，输出结果可以是float或者是int8，视该卷积模块的后面一个op的情况而定。而训练量化的本质就是在训练的过程中去模拟量化操作的影响，借由训练来使得模型学习并适应这种影响，以此来提高最后量化模型的准确率。
因此在两种 FakeQuant 模块中，我们的主要计算为
![image.png](https://cdn.nlark.com/yuque/0/2020/png/405909/1582775538909-a701341d-ced6-48ad-9df3-d90b7d1cca36.png#height=538&id=thJFB&name=image.png&originHeight=1076&originWidth=632&originalType=binary&size=203698&status=done&style=none&width=316)
对于权值和特征的fake-quant基本都和上图一致，不一样的是对于特征由于其范围是随输入动态变化的，而最终int8模型中必须固定一个对于输入特征的scale值，所以，我们对每一此前向计算出来的scale进行了累积更新，例如使用滑动平均，或者直接取每一次的最大值。对于权值的scale，则没有进行平均，因为每一次更新之后的权值都是学习之后的较好的结果，没有状态保留。
此外，对于特征，我们提供了分通道(PerChannel)或者不分通道(PerTensor)的scale统计方法，可根据效果选择使用。对于权值，我们则使用分通道的量化方法，效果较好。

上述是在训练中的training阶段的计算过程，在test阶段，我们会将BatchNorm合进权值，使用训练过程得到的特征scale和此时权值的scale（每次重新计算得到）对特征和权值进行量化，并真实调用MNN中的 _FloatToInt8 和 _Int8ToFloat 来进行推理，以保证测试得到的结果和最后转换得到的全int8推理模型的结果一致。

最后保存模型的时候会自动保存test阶段的模型，并去掉一些冗余的算子，所以直接保存出来即是全int8推理模型。

### 训练量化结果
目前我们在Lenet，MobilenetV2，以及内部的一些人脸模型上进行了测试，均取得了不错的效果，下面给出MobilenetV2的一些详细数据

|  | 准确率 / 模型大小 |
| --- | --- |
| 原始float模型 | 72.324% / 13M |
| MNN训练量化int8模型 | 72.456% / 3.5M |
| TF训练量化int8模型 | 71.1% / 3.5M (原始 71.8% / 13M) |


上述数据是使用batchsize为32，训练100次迭代得到的，即仅使用到了3200张图片进行训练量化，在ImageNet验证集5万张图片上进行测试得到。可以看到int8量化模型的准确率甚至比float还要高一点，而模型大小下降了73%，同时还可以得到推理速度上的增益。

【注】此处使用到的float模型为TensorFlow官方提供的模型，但官方给出的准确率数据是71.8%，我们测出来比他们要高一点，原因是因为我们使用的预处理代码上有细微差别所致。

### 使用训练量化的一些建议

1. 模型转换时保留BatchNorm和Dropout等训练中会用到的算子，这些算子对训练量化也有帮助
2. 要使用原始模型接近收敛阶段的训练参数，训练参数不对，将导致训练量化不稳定
3. 学习率要调到比较小
4. 我们仅对卷积层实现了训练量化，因此如果用MNN从零开始搭建模型，后期接训练量化，或者Finetune之后想继续训练量化，那么需要用卷积层来实现全连接层即可对全连接层也进行训练量化。示例代码如下
```cpp
// 用卷积层实现输入1280，输出为4的全连接层
NN::ConvOption option;
option.channel = {1280, 4};
mLastConv      = std::shared_ptr<Module>(NN::Conv(option));
```

### 训练量化的配置选项
详见 MNN_ROOT/tools/train/source/module/PipelineModule.hpp
```cpp
// 特征scale的计算方法
enum FeatureScaleStatMethod {
    PerTensor = 0, // 对特征不分通道进行量化
    PerChannel = 1 // 对特征分通道进行量化，deprecated
};
// 特征scale的更新方法
enum ScaleUpdateMethod {
    Maximum = 0, // 使用每一次计算得到的scale的最大值
    MovingAverage = 1 // 使用滑动平均来更新
};
// 指定训练量化的bit数，特征scale的计算方法，特征scale的更新方法，
void toTrainQuant(const int bits = 8, NN::FeatureScaleStatMethod featureScaleStatMethod = NN::PerTensor,
                      NN::ScaleUpdateMethod scaleUpdateMethod = NN::MovingAverage);
```

## 使用预训练模型Finetune
### Finetune原理
深度模型可以看做一种特征提取器，例如卷积神经网络（CNN）可以看做一种视觉特征提取器。但是这种特征提取器需要进行广泛的训练才能避免过拟合数据集，取得较好的泛化性能。如果我们直接搭建一个模型然后在我们自己的小数据集上进行训练的话，很容易过拟合。这时候我们就可以用在一些相似任务的大型数据集上训练得到的模型，在我们自己的小数据集上进行Finetune，即可省去大量训练时间，且获得较好的性能表现。

### 使用场景
例如对于图像分类任务，我们可以使用在ImageNet数据集上训练过的模型如MobilenetV2等，取出其特征提取部分，而替换其最后的分类层（ImageNet有1000类，我们自己的数据集可能只有10类），然后仅对替换之后的分类层进行训练即可。这是因为MobilenetV2的特征提取部分已经得到了较充分的训练，这些特征提取器提取出来的特征对于其他图像是通用的。还有很多其他的任务，例如NLP中可以用在大型语料库上训练过的BERT模型在自己的语料库上进行Finetune。

### MNN Finetune示例
下面以MobilenetV2在自己的4分类小数据集上Finetune为例，演示MNN中Finetune的用法。相关代码在 MNN_ROOT/tools/train/source/demo/mobilenetV2Train.cpp 和 MNN_ROOT/tools/train/source/demo/mobilenetV2Utils.cpp中，可以适当选择大一点的学习率如0.001，加快学习速度
注意，此demo中需要MobilenetV2的MNN模型
```cpp
//  mobilenetV2Train.cpp

class MobilenetV2TransferModule : public Module {
public:
    MobilenetV2TransferModule(const char* fileName) {
        // 读取原始MobilenetV2模型
        auto varMap  = Variable::loadMap(fileName);
        // MobilenetV2的输入节点
        auto input   = Variable::getInputAndOutput(varMap).first.begin()->second;
        // MobilenetV2分类层之前的节点，AveragePooling的输出
        auto lastVar = varMap["MobilenetV2/Logits/AvgPool"];

        // 初始化一个4分类的全连接层，MNN中可以用卷积来表示全连接层
        NN::ConvOption option;
        option.channel = {1280, 4};
        mLastConv      = std::shared_ptr<Module>(NN::Conv(option));

        // 初始化内部特征提取器, 内部提取器设成不需要训练
        mFix.reset(PipelineModule::extract({input}, {lastVar}, false));

        // 注意这里只注册了我们新初始化的4分类全连接层，那么训练时将只更新此4分类全连接层
        registerModel({mLastConv});
    }
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        // 输入一张图片，获得MobilenetV2特征提取器的输出
        auto pool   = mFix->forward(inputs[0]);
        
        // 将上面提取的特征输入到新初始化的4分类层进行分类
        auto result = _Softmax(_Reshape(_Convert(mLastConv->forward(pool), NCHW), {0, -1}));
        
        return {result};
    }
    // MobilenetV2特征提取器，从输入一直到最后一个AveragePooling
    std::shared_ptr<Module> mFix;
    // 重新初始化的4分类全连接层
    std::shared_ptr<Module> mLastConv;
};

class MobilenetV2Transfer : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 6) {
            std::cout << "usage: ./runTrainDemo.out MobilentV2Transfer /path/to/mobilenetV2Model path/to/train/images/ path/to/train/image/txt path/to/test/images/ path/to/test/image/txt"
                      << std::endl;
            return 0;
        }

        std::string trainImagesFolder = argv[2];
        std::string trainImagesTxt = argv[3];
        std::string testImagesFolder = argv[4];
        std::string testImagesTxt = argv[5];
        
		// 读取模型，并替换最后一层分类层
        std::shared_ptr<Module> model(new MobilenetV2TransferModule(argv[1]));
		// 进入训练环节
        MobilenetV2Utils::train(model, 4, 0, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt);

        return 0;
    }
};
```

## 蒸馏训练
### 蒸馏训练原理
蒸馏（Distillation）总体思想是将一个模型所学到的知识蒸馏转移到另外一个模型上，就像教师教学生一样，因此前一个模型常被称为教师模型，后面一个模型常被称为学生模型。如果学生模型比教师模型小，那么蒸馏也成为一种模型压缩方法。Hinton在2015年提出了蒸馏这一思想，具体做法可以参考论文：
[Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).](https://arxiv.org/abs/1503.02531)

### MNN蒸馏训练示例
以MobilenetV2的蒸馏训练量化为例，我们来看一下MNN中怎样做蒸馏训练，相关代码在 MNN_ROOT/tools/train/source/demo/distillTrainQuant.cpp中。
根据蒸馏算法，我们需要取出输入到模型Softmax节点的logits，然后加上温度参数，最后计算蒸馏loss进行训练
注意，此demo中需要MobilenetV2的MNN模型
```cpp
// distillTrainQuant.cpp

......

// 读取教师MNN模型
auto varMap      = Variable::loadMap(argv[1]);
if (varMap.empty()) {
    MNN_ERROR("Can not load model %s\n", argv[1]);
    return 0;
}

......

// 获取教师模型的输入输出节点
auto inputOutputs = Variable::getInputAndOutput(varMap);
auto inputs       = Variable::mapToSequence(inputOutputs.first);
MNN_ASSERT(inputs.size() == 1);
// 教师模型的输入节点及其名称
auto input = inputs[0];
std::string inputName = input->name();
auto inputInfo = input->getInfo();
MNN_ASSERT(nullptr != inputInfo && inputInfo->order == NC4HW4);

// 教师模型的输出节点及其名称
auto outputs = Variable::mapToSequence(inputOutputs.second);
std::string originOutputName = outputs[0]->name();

// 教师模型输入到Softmax之前的节点，即logits
std::string nodeBeforeSoftmax = "MobilenetV2/Predictions/Reshape";
auto lastVar = varMap[nodeBeforeSoftmax];
std::map<std::string, VARP> outputVarPair;
outputVarPair[nodeBeforeSoftmax] = lastVar;

// 抽取出从输入一直到logits输出的模型部分
auto logitsOutput = Variable::mapToSequence(outputVarPair);
{
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
}

// 将原始模型输入到logits部分转换为float可训练模型
std::shared_ptr<Module> model(PipelineModule::extract(inputs, logitsOutput, true));

// 将上述模型转换为训练量化模型
PipelineModule::turnQuantize(model.get(), bits);

// 原始模型不会进行训练，只会进行前向推理
std::shared_ptr<Module> originModel(PipelineModule::extract(inputs, logitsOutput, false));
// 进入训练环节
_train(originModel, model, inputName, originOutputName);
```

OK，上面演示了如何获取logits输出，并将模型转换成训练量化模型，下面看一下训练工程中实现量化的关键部分代码
```cpp
// 训练中的一次前向计算过程

// 将输入数据转换成MNN内部的NC4HW4格式
auto nc4hw4example = _Convert(example, NC4HW4);
// 教师模型前向，得到教师模型的logits输出
auto teacherLogits = origin->forward(nc4hw4example);
// 学生模型前向，得到学生模型的logits输出
auto studentLogits = optmized->forward(nc4hw4example);

// 计算label的One-Hot向量
auto labels = trainData[0].second[0];
const int addToLabel = 1;
auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(labels + _Scalar<int32_t>(addToLabel), {})),
                         _Scalar<int>(1001), _Scalar<float>(1.0f),
                         _Scalar<float>(0.0f));
// 使用教师模型和学生模型的logits，以及真实label，来计算蒸馏loss
// 温度T = 20，softTargets loss系数 0.9
VARP loss = _DistillLoss(studentLogits, teacherLogits, newTarget, 20, 0.9);
```

下面来看一下，蒸馏loss是如何计算的，代码在 MNN_ROOT/tools/train/source/optimizer/Loss.cpp中
```cpp
// Loss.cpp

Express::VARP _DistillLoss(Express::VARP studentLogits, Express::VARP teacherLogits, Express::VARP oneHotTargets, const float temperature, const float alpha) {
    auto info = teacherLogits->getInfo();
    if (info->order == NC4HW4) {
        teacherLogits = _Convert(teacherLogits, NCHW);
        studentLogits = _Convert(studentLogits, NCHW);
    }
    MNN_ASSERT(studentLogits->getInfo()->dim.size() == 2);
    MNN_ASSERT(studentLogits->getInfo()->dim == teacherLogits->getInfo()->dim);
    MNN_ASSERT(studentLogits->getInfo()->dim == oneHotTargets->getInfo()->dim);
    MNN_ASSERT(alpha >= 0 && alpha <= 1);
    // 计算考虑温度之后的教师模型softTargets，学生模型的预测输出
    auto softTargets = _Softmax(teacherLogits * _Scalar(1 / temperature));
    auto studentPredict = _Softmax(studentLogits * _Scalar(1 / temperature));
    // 计算softTargets产生的loss
    auto loss1 = _Scalar(temperature * temperature) * _KLDivergence(studentPredict, softTargets);
    // 计算label产生的loss
    auto loss2 = _CrossEntropy(_Softmax(studentLogits), oneHotTargets);
    // 总的loss为两者加权之和
    auto loss = _Scalar(alpha) * loss1 + _Scalar(1 - alpha) * loss2;
    return loss;
}
```