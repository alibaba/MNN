//
//  mobilenetV2Train.cpp
//  MNN
//
//  Created by MNN on 2020/01/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include "DemoUnit.hpp"
#include "MobilenetV2.hpp"
#include "MobilenetV2Utils.hpp"
#include "NN.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "module/PipelineModule.hpp"

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class MobilenetV2TransferModule : public Module {
public:
    MobilenetV2TransferModule(const char* fileName) {
        auto varMap  = Variable::loadMap(fileName);
        auto input   = Variable::getInputAndOutput(varMap).first.begin()->second;
        auto lastVar = varMap["MobilenetV2/Logits/AvgPool"];

        NN::ConvOption option;
        option.channel = {1280, 4};
        mLastConv      = std::shared_ptr<Module>(NN::Conv(option));

        mFix.reset(NN::extract({input}, {lastVar}, false));

        // Only train last parameter
        registerModel({mLastConv});
    }
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        auto pool   = mFix->forward(inputs[0]);
        auto result = _Softmax(_Reshape(_Convert(mLastConv->forward(pool), NCHW), {0, -1}));
        return {result};
    }
    std::shared_ptr<Module> mFix;
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

        std::shared_ptr<Module> model(new MobilenetV2TransferModule(argv[1]));

        MobilenetV2Utils::train(model, 4, 0, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt);

        return 0;
    }
};

class MobilenetV2Train : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 5) {
            std::cout << "usage: ./runTrainDemo.out MobilenetV2Train path/to/train/images/ path/to/train/image/txt path/to/test/images/ path/to/test/image/txt" << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string trainImagesFolder = argv[1];
        std::string trainImagesTxt = argv[2];
        std::string testImagesFolder = argv[3];
        std::string testImagesTxt = argv[4];

        std::shared_ptr<Module> model(new MobilenetV2);

        MobilenetV2Utils::train(model, 1001, 1, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt);

        return 0;
    }
};

class MobilenetV2PostTrain : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 6) {
            std::cout << "usage: ./runTrainDemo.out MobilentV2PostTrain /path/to/mobilenetV2Model path/to/train/images/ path/to/train/image/txt path/to/test/images/ path/to/test/image/txt"
                      << std::endl;
            return 0;
        }

        std::string trainImagesFolder = argv[2];
        std::string trainImagesTxt = argv[3];
        std::string testImagesFolder = argv[4];
        std::string testImagesTxt = argv[5];

        auto varMap = Variable::loadMap(argv[1]);
        if (varMap.empty()) {
            MNN_ERROR("Can not load model %s\n", argv[1]);
            return 0;
        }

        auto inputOutputs = Variable::getInputAndOutput(varMap);
        auto inputs       = Variable::mapToSequence(inputOutputs.first);
        auto outputs      = Variable::mapToSequence(inputOutputs.second);
        std::shared_ptr<Module> model(NN::extract(inputs, outputs, true));

        MobilenetV2Utils::train(model, 1001, 1, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt);

        return 0;
    }
};

class MobilenetV2TrainQuant : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 6) {
            std::cout << "usage: ./runTrainDemo.out MobilentV2TrainQuant /path/to/mobilenetV2Model path/to/train/images/ path/to/train/image/txt path/to/test/images/ path/to/test/image/txt [bits]"
                      << std::endl;
            return 0;
        }

        std::string trainImagesFolder = argv[2];
        std::string trainImagesTxt = argv[3];
        std::string testImagesFolder = argv[4];
        std::string testImagesTxt = argv[5];

        auto varMap = Variable::loadMap(argv[1]);
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

        auto inputOutputs = Variable::getInputAndOutput(varMap);
        auto inputs       = Variable::mapToSequence(inputOutputs.first);
        auto outputs      = Variable::mapToSequence(inputOutputs.second);

        std::shared_ptr<Module> model(NN::extract(inputs, outputs, true));
        NN::turnQuantize(model.get(), bits);

        MobilenetV2Utils::train(model, 1001, 1, trainImagesFolder, trainImagesTxt, testImagesFolder, testImagesTxt);

        return 0;
    }
};

DemoUnitSetRegister(MobilenetV2Transfer, "MobilenetV2Transfer");
DemoUnitSetRegister(MobilenetV2Train, "MobilenetV2Train");
DemoUnitSetRegister(MobilenetV2PostTrain, "MobilenetV2PostTrain");
DemoUnitSetRegister(MobilenetV2TrainQuant, "MobilenetV2TrainQuant");
