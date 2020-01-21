//
//  quanMnist.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Optimizer.hpp>
#include <cmath>
#include <sstream>
#include <vector>
#include "DemoUnit.hpp"
#include "FixModule.hpp"
#include "NN.hpp"
#include "PipelineModule.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <functional>
#include "MNN_generated.h"
#include "MnistUtils.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

class QuanMnist : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 3) {
            MNN_PRINT("usage: ./runTrainDemo.out QuanMnist /path/to/mnistModel /path/to/unzipped/mnist/data/ [bits]\n");
            return 0;
        }
        std::string root = argv[2];
        auto varMap      = Variable::loadMap(argv[1]);
        if (varMap.empty()) {
            MNN_ERROR("Can not load model %s\n", argv[1]);
            return 0;
        }
        int bits = 8;
        if (argc > 3) {
            std::istringstream is(argv[3]);
            is >> bits;
        }
        if (1 > bits || bits > 8) {
            MNN_ERROR("bits must be 2-8, use 8 default\n");
            bits = 8;
        }
        FUNC_PRINT(bits);
        auto inputOutputs = Variable::getInputAndOutput(varMap);
        auto inputs       = Variable::mapToSequence(inputOutputs.first);
        auto outputs      = Variable::mapToSequence(inputOutputs.second);
        std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(EXPRP)> transformFunction =
            [bits](EXPRP source) {
                if (source->get() == nullptr) {
                    return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
                }
                auto convExtracted = NN::Utils::ExtractConvolution(source);
                if (std::get<1>(convExtracted) == nullptr) {
                    return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
                }
                std::shared_ptr<Module> module(NN::ConvInt8(std::get<0>(convExtracted), std::get<1>(convExtracted),
                                                            std::get<2>(convExtracted), std::get<3>(convExtracted),
                                                            bits));
                return std::make_pair(std::vector<int>{0}, module);
            };
        Transformer::turnModelToTrainable(Transformer::TrainConfig())->onExecute(outputs);
        std::shared_ptr<Module> model(new PipelineModule(inputs, outputs, transformFunction));

        MnistUtils::train(model, root);
        return 0;
    }
};
DemoUnitSetRegister(QuanMnist, "QuanMnist");

class OctaveMnist : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 3) {
            MNN_PRINT("usage: ./runTrainDemo.out OctaveMnist /path/to/mnistModel /path/to/unzipped/mnist/data/ \n");
            return 0;
        }
        std::string root = argv[2];
        auto varMap      = Variable::loadMap(argv[1]);
        if (varMap.empty()) {
            MNN_ERROR("Can not load model %s\n", argv[1]);
            return 0;
        }
        auto inputOutputs = Variable::getInputAndOutput(varMap);
        auto inputs       = Variable::mapToSequence(inputOutputs.first);
        auto outputs      = Variable::mapToSequence(inputOutputs.second);
        std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(EXPRP)> transformFunction =
            [](EXPRP source) {
                if (source->get() == nullptr) {
                    return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
                }
                auto convExtracted = NN::Utils::ExtractConvolution(source);
                if (std::get<1>(convExtracted) == nullptr) {
                    return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
                }
                if (std::get<0>(convExtracted).channel[0] <= 4 || std::get<0>(convExtracted).channel[1] <= 4) {
                    return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
                }
                std::shared_ptr<Module> module(NN::ConvOctave(std::get<0>(convExtracted), std::get<1>(convExtracted),
                                                              std::get<2>(convExtracted), std::get<3>(convExtracted),
                                                              0.5f, 0.5f));
                return std::make_pair(std::vector<int>{0}, module);
            };
        Transformer::turnModelToTrainable(Transformer::TrainConfig())->onExecute(outputs);
        std::shared_ptr<Module> model(new PipelineModule(inputs, outputs, transformFunction));

        MnistUtils::train(model, root);
        return 0;
    }
};
DemoUnitSetRegister(OctaveMnist, "OctaveMnist");
