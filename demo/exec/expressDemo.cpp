//
//  expressDemo.cpp
//  MNN
//
//  Created by MNN on b'2019/08/19'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace MNN::Express;
#define UP_DIV(x) (((x)+3)/4)

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        MNN_ERROR("./expressDemo.out model_path type testTime\n");
        return 0;
    }
    auto modelFileName = argv[1];
    FUNC_PRINT_ALL(modelFileName, s);
    auto exe = Executor::getGlobalExecutor();
    MNN::BackendConfig config;
    config.precision = MNN::BackendConfig::Precision_Low;
    MNNForwardType forwardType = MNN_FORWARD_CPU;
    if (argc >= 3) {
        forwardType = (MNNForwardType)atoi(argv[2]);
    }
    exe->setGlobalExecutorConfig(forwardType, config, 4);
    auto model = Variable::loadMap(modelFileName);
    auto inputOutput = Variable::getInputAndOutput(model);
    auto inputs = inputOutput.first;
    auto outputs = inputOutput.second;
    int testTime = 10;
    if (argc >= 4) {
        testTime = atoi(argv[3]);
    }
    auto input = inputs.begin()->second;
    auto output = outputs.begin()->second;
    //input->resize({1, 224, 224, 3});
    auto inputInfo = input->getInfo();
    if (nullptr == inputInfo) {
        return 0;
    }
    {
        AUTOTIME;
        input = _ChangeInputFormat(input, NCHW);
        inputInfo = input->getInfo();
        if (output->getInfo()->order == NC4HW4) {
            output = _Convert(output, NCHW);
        }
    }
    auto outputInfo = output->getInfo();
    if (nullptr == outputInfo) {
        MNN_ERROR("Output Not valid\n");
        return 0;
    }
    //Test Speed
    if (testTime > 0){
        //Let the frequence up
        for (int i=0; i<3; ++i) {
            input->writeMap<float>();
            input->unMap();
            output->readMap<float>();
            output->unMap();
        }
        AUTOTIME;
        for (int i=0; i<testTime; ++i) {
            input->writeMap<float>();
            input->unMap();
            output->readMap<float>();
            output->unMap();
        }
    }
    {
        auto size = inputInfo->size;
        auto inputPtr = input->writeMap<float>();
        std::ifstream inputOs("input_0.txt");
        for (int i=0; i<size; ++i) {
            inputOs >> inputPtr[i];
        }
        input->unMap();
    }

    {
        auto size = outputInfo->size;
        auto outputPtr = output->readMap<float>();
        if (nullptr == outputPtr) {
            MNN_ERROR("Output Not valid read error\n");
            return 0;
        }
        std::ofstream outputOs("output.txt");
        for (int i=0; i<size; ++i) {
            outputOs << outputPtr[i] << "\n";
        }
        output->unMap();
    }

    return 0;
}
