//
//  CloneNetTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <MNN/expr/Module.hpp>
#include "TestUtils.h"

using namespace MNN;
using namespace MNN::Express;

const int width = 1280;
const int height = 720;
const int channel = 3;
static std::shared_ptr<Module> _createModel() {
    float fac = 0.23;
    int res = 10;
    float tail = 0.05;
    std::vector<float> constdata(channel * height * width);
    for (int j = 0; j < channel; ++j) {
        for (int k = 0; k < height * width; k++) {
            constdata[j * height * width + k] = (j * height * width + k) % (height * width) * fac + tail;
        }
    }
    
    auto x = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
    x->setName("Input");
    auto c = _Const(constdata.data(), {1, channel, height, width}, NCHW);
    auto y = x + c;
    y->setName("Output");
    std::unique_ptr<NetT> net(new NetT);
    Variable::save({y}, net.get());
    flatbuffers::FlatBufferBuilder builder;
    auto len = MNN::Net::Pack(builder, net.get());
    builder.Finish(len);
    return std::shared_ptr<Module>(Module::load({"Input"}, {"Output"}, builder.GetBufferPointer(), builder.GetSize()));
}

class CloneNetTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        std::vector<float> inputData(channel * width * height);
        for (int i = 0; i < channel * height * width; ++i){
            inputData[i] = (rand() % 10) * 0.1;
        }
        
        auto net = _createModel();
        auto x = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
        {
            auto xPtr = x->writeMap<float>();
            ::memcpy(xPtr, inputData.data(), channel * height * width * sizeof(float));
            x->unMap();
        }
        
        auto outputs = net->onForward({x});
        outputs[0] = _Convert(outputs[0], NC4HW4);
        auto refPtr = outputs[0]->readMap<float>();
        auto size = outputs[0]->getInfo()->size;
        
        
        // clone model
        MNN::BackendConfig config;
        config.precision = (MNN::BackendConfig::PrecisionMode)MNN::BackendConfig::Precision_Normal;
        config.memory = (MNN::BackendConfig::MemoryMode)MNN::BackendConfig::Memory_Normal;
        std::shared_ptr<Executor> executor(Executor::newExecutor(getCurrentType(), config, 4));
        ExecutorScope scope(executor);
        std::unique_ptr<Module> tempModule(Module::clone(net.get()));
        
        auto xClone = _Input({1, channel, height, width}, NCHW, halide_type_of<float>());
        {
            auto xPtr = xClone->writeMap<float>();
            ::memcpy(xPtr, inputData.data(), channel * height * width * sizeof(float));
            xClone->unMap();
        }
        auto outputsClone = tempModule->onForward({xClone});
        outputsClone[0] = _Convert(outputsClone[0], NC4HW4);
        auto outPtr = outputsClone[0]->readMap<float>();
        
        for (int i = 0; i < size; ++i) {
            float targetValue = refPtr[i], computeResult = outPtr[i];
            float diff = targetValue - computeResult;
            if (fabsf(diff) > 0.001) {
                MNN_PRINT("%d result Error: right=%f, error=%f\n", targetValue, computeResult);
                return false;
            }
        }
        
        return true;
    }
};

MNNTestSuiteRegister(CloneNetTest, "Clone/CloneNet");
