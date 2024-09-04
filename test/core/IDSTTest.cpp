//
//  IDSTTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include <MNN/Tensor.hpp>
#include "core/IDSTEncoder.hpp"
#include "core/ConvolutionCommon.hpp"
using namespace MNN;

class IDSTTest : public MNNTestCase {
public:
    virtual ~IDSTTest() = default;
    virtual bool run(int precision) {
        int kernelNum = 2, kernelSize = 8;
        std::vector<float> weight(kernelNum * kernelSize, 0.f);
        std::vector<float> scale(kernelNum, 0.f);
        std::vector<int8_t> quantWeight(kernelNum * kernelSize, 0);
        // IDST encode
        std::unique_ptr<IDSTQuanT> idstQuantT = IDSTEncoder::encode(weight.data(), scale, kernelSize, kernelNum, false, quantWeight.data(), -127);
        Convolution2DT* conv2dT = new Convolution2DT;
        std::unique_ptr<OpT> opT(new OpT);
        conv2dT->quanParameter = std::move(idstQuantT);
        opT->type = OpType_Convolution;
        opT->main.type = OpParameter_Convolution2D;
        opT->main.value = conv2dT;
        flatbuffers::FlatBufferBuilder builder;
        auto lastOffset = Op::Pack(builder, opT.get());
        builder.Finish(lastOffset);
        auto op = flatbuffers::GetRoot<Op>(builder.GetBufferPointer());
        // IDST decode
        std::shared_ptr<ConvolutionCommon::Int8Common> common = ConvolutionCommon::load(op);
        // is input == output ?
        bool res = (0 == memcmp(common->weightFloat.get(), weight.data(), weight.size()));
        return res;
    }
};
MNNTestSuiteRegister(IDSTTest, "core/idst");
