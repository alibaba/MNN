//
//  MnistUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MnistUtils.hpp"
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include "MnistDataset.hpp"
#include "NN.hpp"
#include "SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

void MnistUtils::train(std::shared_ptr<Module> model, std::string root) {
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 2);
    std::shared_ptr<SGD> sgd(new SGD);
    sgd->append(model->parameters());
    sgd->setMomentum(0.9f);
    // sgd->setMomentum2(0.99f);
    sgd->setWeightDecay(0.0005f);

    auto dataset = std::make_shared<MnistDataset>(root, MnistDataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    auto transform = std::make_shared<StackTransform>();

    const size_t batchSize  = 64;
    const size_t numWorkers = 4;
    bool shuffle            = true;

    auto dataLoader = DataLoader::makeDataLoader(dataset, {transform}, batchSize, shuffle, numWorkers);

    const size_t iterations = dataset->size() / batchSize;

    auto testDataset            = std::make_shared<MnistDataset>(root, MnistDataset::Mode::TEST);
    const size_t testBatchSize  = 20;
    const size_t testNumWorkers = 1;
    shuffle                     = false;

    auto testDataLoader = DataLoader::makeDataLoader(testDataset, {transform}, testBatchSize, shuffle, testNumWorkers);

    const size_t testIterations = testDataset->size() / testBatchSize;

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            for (int i = 0; i < iterations; i++) {
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto cast       = _Cast<float>(example.data[0]);
                example.data[0] = cast * _Const(1.0f / 255.0f);

                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(example.target[0]), _Scalar<int>(10), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

                auto predict = model->forward(example.data[0]);
                auto loss    = _CrossEntropy(predict, newTarget);
                float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);
                if ((epoch * iterations + i) % 100 == 0) {
                    std::cout << "train iteration: " << epoch * iterations + i;
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate << std::endl;
                }
                sgd->step(loss);
                if (i == iterations - 1) {
                    model->setIsTraining(false);
                    auto forwardInput = _Input({1, 1, 28, 28}, NCHW);
                    forwardInput->setName("data");
                    predict = model->forward(forwardInput);
                    predict->setName("prob");
                    Variable::save({predict}, "temp.mnist.mnn");
                }
            }
        }

        int correct = 0;
        testDataLoader->reset();
        model->setIsTraining(false);
        for (int i = 0; i < testIterations; i++) {
            exe->gc(Executor::PART);
            if ((i + 1) % 100 == 0) {
                std::cout << "test iteration: " << (i + 1) << std::endl;
            }
            auto data       = testDataLoader->next();
            auto example    = data[0];
            auto cast       = _Cast<float>(example.data[0]);
            example.data[0] = cast * _Const(1.0f / 255.0f);
            auto predict    = model->forward(example.data[0]);
            predict         = _ArgMax(predict, 1);
            auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.target[0]))).sum({});
            correct += accu->readMap<int32_t>()[0];
        }
        auto accu = (float)correct / (float)testDataset->size();
        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
        exe->dumpProfile();
    }
}
