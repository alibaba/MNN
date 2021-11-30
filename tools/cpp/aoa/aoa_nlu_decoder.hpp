//
//  aoa_nlu_decoder.cpp
//  MNN
//
//  Created by MNN on b'2021/09/06'.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited
//

#include <MNN/expr/Module.hpp>
// #define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "../../tools/cpp/revertMNNModel.hpp"
#include <fstream>
#include <sstream>
#include <stdio.h>
#include<string.h>
using namespace MNN::Express;
using namespace MNN;
using namespace std;

class AOANLUDecoder
{
public:
    AOANLUDecoder() {};
    ~AOANLUDecoder() {};

    virtual void getInputOutput(std::string& outputTensorName, std::vector<std::string>& inputTensorName, std::vector<std::vector<int>>& inputTensorShape, const int sequenceLength) = 0;

    int run(int argc, const char* argv[]) {
        if (argc < 2) {
            MNN_ERROR("Don't has model name\n");
                return 0;
        }
        int loop = 1;
        int warmup = 1;
        int sequenceLength = 128;
        float sparsity = 0.0f;
        int sparseBlockOC = 1;
        if (argc >= 3) {
            loop = atoi(argv[2]);
        }
        if (argc >= 4) {
            warmup = atoi(argv[3]);
        }
        if (argc >= 5) {
            sequenceLength = atoi(argv[4]);
        }

        if(argc >= 6) {
            sparsity = atof(argv[5]);
        }

        if(argc >= 7) {
            sparseBlockOC = atoi(argv[6]);
        }
        std::string modelName = argv[1];

        runNLU(modelName, loop, warmup, sequenceLength, sparsity, sparseBlockOC);

        return 0;
    }

    void runNLU(std::string modelName, int loop, int warmup, int sequenceLength, float sparsity, int sparseBlockOC) {

        MNN::BackendConfig backendConfig;
        backendConfig.precision = MNN::BackendConfig::Precision_Normal;
        backendConfig.power = MNN::BackendConfig::Power_High;

        MNN::ScheduleConfig config;
        config.numThread = 1;
        config.type      = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
        config.backendConfig = &backendConfig;
        Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU, backendConfig, 1);

        std::shared_ptr<Module> model;

        std::string outputTensorName;
        std::vector<std::string> inputTensorName;
        std::vector<std::vector<int>> inputTensorShape;
        getInputOutput(outputTensorName, inputTensorName, inputTensorShape, sequenceLength);


        std::vector<Express::VARP> inputTensors;
        for (int i = 0; i < inputTensorShape.size(); ++i) {
            Express::VARP input = _Input(inputTensorShape[i], NHWC, halide_type_of<float>());
            // MNN_PRINT("%d th, size:%d\n", i, input->getInfo()->size);
            ::memset(input->writeMap<void>(), 0, input->getInfo()->size);
            inputTensors.emplace_back(input);
        }
        Express::VARP start_tokens = _Input({1}, NHWC, halide_type_of<int>());
        ::memset(start_tokens->writeMap<void>(), 0, start_tokens->getInfo()->size);
        inputTensors.emplace_back(start_tokens);


        // model.reset(Module::load(inputTensorName, {outputTensorName}, modelName.c_str()));

        auto revertor = std::unique_ptr<Revert>(new Revert(modelName.c_str()));
        revertor->initialize(sparsity, sparseBlockOC, true);
        auto modelBuffer      = reinterpret_cast<const uint8_t*>(revertor->getBuffer());
        const auto bufferSize = revertor->getBufferSize();
        model.reset(Module::load(inputTensorName, {outputTensorName}, modelBuffer, bufferSize));
        revertor.reset();

        std::vector<VARP> outputs;
        for (int i = 0; i < warmup; ++i) {
            {
                Executor::getGlobalExecutor()->resetProfile();
                outputs = model->onForward(inputTensors);
                Executor::getGlobalExecutor()->dumpProfile();
            }

            // std::ostringstream fileNameOs;
            // std::ostringstream dimInfo;
            // fileNameOs << i << "_output.txt";
            // auto info = outputs[0]->getInfo();
            // for (int d=0; d<info->dim.size(); ++d) {
            //     dimInfo << info->dim[d] << "_";
            // }
            // auto fileName = fileNameOs.str();
            // MNN_PRINT("Output Name: %s, Dim: %s\n", fileName.c_str(), dimInfo.str().c_str());
            // auto ptr = outputs[0]->readMap<float>();
            // std::ofstream outputOs(fileName.c_str());
            // for (int i=0; i<info->size; ++i) {
            //     outputOs << ptr[i] << "\n";
            // }
        }

        Timer timer;
        timer.reset();
        for (int i = 0; i < loop; ++i) {
            outputs = model->onForward(inputTensors);
        }
        float averageTime =  (timer.durationInUs() / 1000.0f) / loop;
        MNN_PRINT("benchmark sparsed aoa v3: sparsity:%.3f, block:1x%d, warmup loops:%d, run loops:%d\nname = %s, avg = %.3f ms\n",
            sparsity, sparseBlockOC, warmup, loop, modelName.c_str(), averageTime);

        return;
    }
};

