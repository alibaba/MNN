# 优化器使用
## SGD with momentum
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

## ADAM
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

## Loss
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