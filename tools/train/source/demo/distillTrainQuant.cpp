//
//  distillTrainQuant.cpp
//  MNN
//
//  Created by MNN on 2020/02/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include "DemoUnit.hpp"
#include "NN.hpp"
#include "SGD.hpp"
#include "module/PipelineModule.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <functional>
#include "RandomGenerator.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "Transformer.hpp"
#include "DataLoader.hpp"
#include "ImageDataset.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace MNN::CV;

std::string gTrainImagePath;
std::string gTrainTxt;
std::string gTestImagePath;
std::string gTestTxt;

void _test(std::shared_ptr<Module> optmized, const ImageDataset::ImageConfig* config) {
    bool readAllImagesToMemory = false;
    DatasetPtr dataset = ImageDataset::create(gTestImagePath, gTestTxt, config, readAllImagesToMemory);

    const int batchSize = 10;
    const int numWorkers = 0;
    std::shared_ptr<DataLoader> dataLoader(dataset.createLoader(batchSize, true, false, numWorkers));

    const int  iterations = dataLoader->iterNumber();

    // const int usedSize = 1000;
    // const int iterations = usedSize / batchSize;

    int correct = 0;
    dataLoader->reset();
    optmized->setIsTraining(false);

    AUTOTIME;
    for (int i = 0; i < iterations; i++) {
        if ((i + 1) % 10 == 0) {
            std::cout << "test iteration: " << (i + 1) << std::endl;
        }
        auto data       = dataLoader->next();
        auto example    = data[0];
        auto predict    = optmized->forward(_Convert(example.first[0], NC4HW4));
        predict = _Softmax(predict);
        predict         = _ArgMax(predict, 1); // (N, numClasses) --> (N)
        const int addToLabel = 1;
        auto label = example.second[0] + _Scalar<int32_t>(addToLabel);
        auto accu       = _Cast<int32_t>(_Equal(predict, label).sum({}));

        correct += accu->readMap<int32_t>()[0];
    }
    auto accu = (float)correct / dataLoader->size();
    // auto accu = (float)correct / usedSize;
    std::cout << "accuracy: " << accu << std::endl;
}

void _train(std::shared_ptr<Module> origin, std::shared_ptr<Module> optmized, std::string inputName, std::string outputName) {
    std::shared_ptr<SGD> sgd(new SGD(optmized));
    sgd->setMomentum(0.9f);
    sgd->setWeightDecay(0.00004f);

    auto converImagesToFormat  = CV::RGB;
    int resizeHeight           = 224;
    int resizeWidth            = 224;
    std::vector<float> means = {127.5, 127.5, 127.5};
    std::vector<float> scales = {1/127.5, 1/127.5, 1/127.5};
    std::vector<float> cropFraction = {0.875, 0.875}; // center crop fraction for height and width
    bool centerOrRandomCrop = false; // true for random crop
    std::shared_ptr<ImageDataset::ImageConfig> datasetConfig(ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight, resizeWidth, scales, means, cropFraction, centerOrRandomCrop));
    bool readAllImagesToMemory = false;
    DatasetPtr dataset = ImageDataset::create(gTrainImagePath, gTrainTxt, datasetConfig.get(), readAllImagesToMemory);

    const int batchSize = 32;
    const int numWorkers = 4;
    auto dataLoader = dataset.createLoader(batchSize, true, true, numWorkers);

    const int iterations = dataLoader->iterNumber();

    for (int epoch = 0; epoch < 5; ++epoch) {
        AUTOTIME;
        dataLoader->reset();
        optmized->setIsTraining(true);
        origin->setIsTraining(false);
        Timer _100Time;
        int lastIndex = 0;
        int moveBatchSize = 0;
        for (int i = 0; i < iterations; i++) {
            // AUTOTIME;
            auto trainData  = dataLoader->next();
            auto example    = trainData[0].first[0];
            moveBatchSize += example->getInfo()->dim[0];
            auto nc4hw4example = _Convert(example, NC4HW4);
            auto teacherLogits = origin->forward(nc4hw4example);
            auto studentLogits = optmized->forward(nc4hw4example);

            // Compute One-Hot
            auto labels = trainData[0].second[0];
            const int addToLabel = 1;
            auto newTarget = _OneHot(_Cast<int32_t>(_Squeeze(labels + _Scalar<int32_t>(addToLabel), {})),
                                     _Scalar<int>(1001), _Scalar<float>(1.0f),
                                     _Scalar<float>(0.0f));

            VARP loss = _DistillLoss(studentLogits, teacherLogits, newTarget, 20, 0.9);

            // float rate   = LrScheduler::inv(basicRate, epoch * iterations + i, 0.0001, 0.75);
            float rate = 1e-5;
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
            sgd->step(loss);
        }

        {
            AUTOTIME;
            dataLoader->reset();
            optmized->setIsTraining(false);
            {
                auto forwardInput = _Input({1, 3, 224, 224}, NC4HW4);
                forwardInput->setName(inputName);
                auto predict = optmized->forward(forwardInput);
                auto output = _Softmax(predict);
                output->setName(outputName);
                Transformer::turnModelToInfer()->onExecute({output});
                Variable::save({output}, "temp.quan.mnn");
            }
        }

        _test(optmized, datasetConfig.get());
    }
}

class DistillTrainQuant : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 6) {
            MNN_PRINT("usage: ./runTrainDemo.out DistillTrainQuant /path/to/mobilenetV2Model path/to/train/images/ path/to/train/image/txt path/to/test/images/ path/to/test/image/txt [bits]\n");
            return 0;
        }

        gTrainImagePath = argv[2];
        gTrainTxt = argv[3];
        gTestImagePath = argv[4];
        gTestTxt = argv[5];

        auto varMap      = Variable::loadMap(argv[1]);
        if (varMap.empty()) {
            MNN_ERROR("Can not load model %s\n", argv[1]);
            return 0;
        }
        int bits = 8;
        if (argc > 6) {
            std::istringstream is(argv[6]);
            is >> bits;
        }
        if (1 > bits || bits > 8) {
            MNN_ERROR("bits must be 2-8, use 8 default\n");
            bits = 8;
        }
        FUNC_PRINT(bits);

        auto inputOutputs = Variable::getInputAndOutput(varMap);
        auto inputs       = Variable::mapToSequence(inputOutputs.first);
        MNN_ASSERT(inputs.size() == 1);
        auto input = inputs[0];
        std::string inputName = input->name();
        auto inputInfo = input->getInfo();
        MNN_ASSERT(nullptr != inputInfo && inputInfo->order == NC4HW4);

        auto outputs = Variable::mapToSequence(inputOutputs.second);
        std::string originOutputName = outputs[0]->name();

        std::string nodeBeforeSoftmax = "MobilenetV2/Predictions/Reshape";
        auto lastVar = varMap[nodeBeforeSoftmax];
        std::map<std::string, VARP> outputVarPair;
        outputVarPair[nodeBeforeSoftmax] = lastVar;

        auto logitsOutput = Variable::mapToSequence(outputVarPair);
        {
            auto exe = Executor::getGlobalExecutor();
            BackendConfig config;
            exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
        }
        std::shared_ptr<Module> model(NN::extract(inputs, logitsOutput, true));
        NN::turnQuantize(model.get(), bits);
        std::shared_ptr<Module> originModel(NN::extract(inputs, logitsOutput, false));
        _train(originModel, model, inputName, originOutputName);
        return 0;
    }
};

DemoUnitSetRegister(DistillTrainQuant, "DistillTrainQuant");
