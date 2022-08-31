# 使用表达式接口训练
[从源码编译](../compile/tools.html#id3)
## 获取可训练模型
首先我们需要获得一个可训练的模型结构，而在MNN中可以用两种方式达到这一目的。

一、将其他框架，如TensorFlow，Pytorch训练得到的模型转成MNN可训练模型，这一过程可使用 MNNConverter 工具实现。典型的应用场景为 1. 使用MNN Finetune，2. 使用MNN进行训练量化。在模型转换过程中建议使用 MNNConverter 的 --forTraining 选项，保留BatchNorm，Dropout等训练过程中会用到的算子。

二、使用MNN从零开始搭建一个模型，并使用MNN进行训练，这可以省去模型转换的步骤，并且也可以十分容易地转换为训练量化模型。在 MNN_ROOT/tools/train/source/models/ 目录中我们提供了Lenet，MobilenetV1，MobilenetV2等使用MNN训练框架搭建的模型。

### 将其他框架模型转换为MNN可训练模型
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

### 使用MNN从零开始搭建模型
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

## 实现数据集接口
这部分在MNN文档中[加载训练数据](train.html#id8) 部分有详细描述。

## 训练并保存模型
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

## 保存和恢复模型
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