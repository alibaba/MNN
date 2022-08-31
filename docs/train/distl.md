# 蒸馏训练
## 蒸馏训练原理
蒸馏（Distillation）总体思想是将一个模型所学到的知识蒸馏转移到另外一个模型上，就像教师教学生一样，因此前一个模型常被称为教师模型，后面一个模型常被称为学生模型。如果学生模型比教师模型小，那么蒸馏也成为一种模型压缩方法。Hinton在2015年提出了蒸馏这一思想，具体做法可以参考论文：
[Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).](https://arxiv.org/abs/1503.02531)

## MNN蒸馏训练示例
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