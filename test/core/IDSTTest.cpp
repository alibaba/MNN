//
//  IDSTTest.cpp
//  MNNTests
//
//  Created by MNN on 2021/04/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include <cmath>
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

class IDSTFp16MetadataTest : public MNNTestCase {
    static bool checkCase(bool asymmetric, int aMin, float quantScale) {
        const int blockSize = 8, blockCount = 4;
        std::vector<float> alpha(asymmetric ? 2 * blockCount : blockCount);
        std::vector<int8_t> codes(blockSize * blockCount);
        for (int b = 0; b < blockCount; ++b) {
            const float scale = 0.0137f + 0.0031f * b;
            if (asymmetric) {
                alpha[2 * b] = -0.1937f + 0.0413f * b + (aMin == 1 ? 128 * scale : 0);
                alpha[2 * b + 1] = scale;
            } else {
                alpha[b] = scale;
            }
        }
        for (int i = 0; i < codes.size(); ++i) {
            codes[i] = (i * 7) % 256 - 128;
        }
        OpT opT;
        opT.type = OpType_Convolution;
        opT.main.type = OpParameter_Convolution2D;
        opT.main.value = new Convolution2DT;
        auto conv = opT.main.AsConvolution2D();
        conv->quanParameter = IDSTEncoder::encode(nullptr, alpha, blockSize, blockCount, asymmetric, codes.data(),
                                                 aMin, {8, false, 16});
        auto quan = conv->quanParameter.get();
        quan->quantScale = quantScale;
        MNNTEST_ASSERT(quan->scaleStorage == ScaleStorageType_FP16 && quan->alpha.empty());
        MNNTEST_ASSERT(quan->alphaFp16.size() == alpha.size());
        // The oracle starts from serialized half bits, not the unrounded source metadata.
        for (int i = 0; i < alpha.size(); ++i) {
            half_float::half h;
            ::memcpy(&h, &quan->alphaFp16[i], sizeof(uint16_t));
            alpha[i] = float(h);
        }
        if (asymmetric && aMin <= 0) {
            for (int b = 0; b < blockCount; ++b) {
                alpha[2 * b] -= (aMin == 0 ? -128 : aMin) * alpha[2 * b + 1];
            }
        }
        for (auto& v : alpha) {
            v *= quantScale;
        }
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(Op::Pack(builder, &opT));
        auto op = flatbuffers::GetRoot<Op>(builder.GetBufferPointer());
        auto lazy = ConvolutionCommon::load(op, nullptr, false, true, nullptr, true);
        auto eager = ConvolutionCommon::load(op, nullptr, false, true, nullptr);
        const bool halfOnly = (!asymmetric || aMin > 0) && quantScale == 1.0f;
        MNNTEST_ASSERT(lazy && eager && lazy->alphaIsFp16 && lazy->alphaHalf.get());
        MNNTEST_ASSERT(lazy->alphaSize == alpha.size() && eager->alphaSize == alpha.size());
        MNNTEST_ASSERT((lazy->alpha.get() == nullptr) == halfOnly && eager->alpha.get());
        MNNTEST_ASSERT(lazy->asymmetric == asymmetric && eager->asymmetric == asymmetric);
        MNNTEST_ASSERT(0 == ::memcmp(lazy->alphaHalf.get(), quan->alphaFp16.data(), alpha.size() * sizeof(uint16_t)));
        MNNTEST_ASSERT(lazy->weight.get() && eager->weight.get());
        MNNTEST_ASSERT(lazy->weight.size() == codes.size() && eager->weight.size() == codes.size());
        MNNTEST_ASSERT(0 == ::memcmp(lazy->weight.get(), codes.data(), codes.size()));
        MNNTEST_ASSERT(0 == ::memcmp(eager->weight.get(), codes.data(), codes.size()));
        for (int i = 0; i < alpha.size(); ++i) {
            MNNTEST_ASSERT(eager->alpha.get()[i] == alpha[i]);
            MNNTEST_ASSERT(halfOnly || lazy->alpha.get()[i] == alpha[i]);
        }
        auto floats = ConvolutionCommon::load(op, nullptr, true, false, nullptr, true);
        MNNTEST_ASSERT(floats && floats->weightFloat.get() && floats->weightFloat.size() == codes.size());
        for (int i = 0; i < codes.size(); ++i) {
            const int b = i / blockSize;
            const float expected = asymmetric ? codes[i] * alpha[2 * b + 1] + alpha[2 * b] : codes[i] * alpha[b];
            MNNTEST_ASSERT(std::isfinite(floats->weightFloat.get()[i]));
            MNNTEST_ASSERT(std::fabs(floats->weightFloat.get()[i] - expected) < 1e-6f);
        }
        return true;
    }

public:
    bool run(int precision) override {
        MNNTEST_ASSERT(checkCase(false, 1, 1.0f));
        MNNTEST_ASSERT(checkCase(true, 1, 1.0f));
        MNNTEST_ASSERT(checkCase(true, -128, 1.0f));
        MNNTEST_ASSERT(checkCase(true, 0, 1.0f));
        MNNTEST_ASSERT(checkCase(false, 1, 1.7f));
        MNNTEST_ASSERT(checkCase(true, 1, 1.7f));
        MNNTEST_ASSERT(checkCase(true, -128, 1.7f));
        return true;
    }
};
MNNTestSuiteRegister(IDSTFp16MetadataTest, "core/idst/fp16_metadata");
