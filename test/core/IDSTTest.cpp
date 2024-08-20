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
        std::unique_ptr<Convolution2DT> conv2dT(new Convolution2DT);
        conv2dT->quanParameter = std::move(idstQuantT);
        flatbuffers::FlatBufferBuilder builder;
        auto lastOffset = Convolution2D::Pack(builder, conv2dT.get());
        builder.Finish(lastOffset);
        auto conv2d = flatbuffers::GetRoot<Convolution2D>(builder.GetBufferPointer());
        // IDST decode
        std::shared_ptr<ConvolutionCommon::Int8Common> common = ConvolutionCommon::load(conv2d);
        // is input == output ?
        bool res = (0 == memcmp(common->weightFloat.get(), weight.data(), weight.size()));
        return res;
    }
};
MNNTestSuiteRegister(IDSTTest, "core/idst");
