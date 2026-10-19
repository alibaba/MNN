//
//  LayerNormTest.cpp
//  MNNTests
//
//  Created by MNN on 2023/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <algorithm>
#include <cmath>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "core/TensorUtils.hpp"

using namespace MNN;
using namespace MNN::Express;

static VARP _LayerNorm(VARP x, std::vector<int32_t> axis, float epsilon, std::vector<float> gamma, std::vector<float> beta, int group = 1, bool useRMS = false) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                         = OpParameter_LayerNorm;
    op->type                              = OpType_LayerNorm;
    op->main.value                        = new LayerNormT;
    if(gamma.size() != 0){
        op->main.AsLayerNorm()->gamma         = gamma;
    }
    if(beta.size() != 0){
        op->main.AsLayerNorm()->beta          = beta;
    }
    op->main.AsLayerNorm()->epsilon       = epsilon;
    op->main.AsLayerNorm()->axis          = axis;
    op->main.AsLayerNorm()->group         = group;
    op->main.AsLayerNorm()->useRMSNorm    = useRMS;
    return (Variable::create(Expr::create(std::move(op), {x})));
}

std::vector<float> inputdata_0 = {0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6,
                                8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6,
                                6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6,
                                4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6,
                                2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6,
                                0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6,
                                8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6,
                                6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6,
                                4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6,
                                2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6,
                                0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6,
                                8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6,
                                6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 0.6, 1.6, 2.6, 3.6,
                                4.6, 5.6, 6.6, 7.6, 8.6, 9.6};
std::vector<float> tgdata_0 = {-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079,
                                -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079, -1.34164079,
                                -0.4472136 ,  0.4472136 ,  1.34164079,  0.86824314,  1.11631261,
                                -1.11631261, -0.86824314, -1.34164079, -0.4472136 ,  0.4472136 ,
                                 1.34164079, -1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079};
std::vector<float> inputdata_1 = {0.7742,  0.5332, -0.8799, -1.0676, -0.7402, -1.5074,  0.2052,  0.3648,
                                  1.5056, -0.2804,  1.2785,  1.3410,  0.5395,  0.1665, -1.4594,  0.1158,
                                  -1.8436, -0.1581, -1.5055,  0.3366,  0.4938,  0.0630,  0.5465,  0.9264,
                                  -1.0491,  2.4297,  1.9942,  0.4256,  1.3902, -0.1021, -0.9883,  0.4500};
std::vector<float> tgdata_1 = {1.1381,  0.8445, -0.8770, -1.1056, -0.4238, -1.4374,  0.8252,  1.0360,
                               0.7544, -1.7206,  0.4397,  0.5264,  0.9098,  0.4242, -1.6923,  0.3583,
                               -1.1587,  0.6996, -0.7859,  1.2451, -0.0446, -1.4518,  0.1277,  1.3688,
                               -1.4550,  1.0769,  0.7599, -0.3818,  1.3930, -0.3354, -1.3618,  0.3041};
float eps = 0.000001f;

static bool testKernel (std::vector<float> inputdata, std::vector<float> targetdata, std::vector<int> dimensions, std::vector<int> reduceAxis, float epsilon, std::vector<float> gamma, std::vector<float> beta, std::vector<float> inputQuan, std::vector<float> outputQuan, std::string testName, int precision, int group = 1) {
    int size = 1;
    for (int i = 0; i < dimensions.size(); ++i) {
        size *= dimensions[i];
    }
    int reducesize = 1;
    for (int i = 0; i < reduceAxis.size(); ++i) {
        reducesize *= dimensions[reduceAxis[i]];
    }
    MNN_ASSERT(gamma.size() == 0 || (gamma.size() >0 && reducesize == gamma.size()));
    MNN_ASSERT(gamma.size() == beta.size());
    auto input = _Input(dimensions, NCHW);
    if (inputQuan.size() > 0) {
        input->writeScaleMap(inputQuan[0], inputQuan[1]);
    }
    auto inputPtr = input->writeMap<float>();
    ::memcpy(inputPtr, inputdata.data(), input->getInfo()->size * sizeof(float));
    auto output = _LayerNorm(input, reduceAxis, epsilon, gamma, beta, group, false);
    if (outputQuan.size() > 0) {
        output->writeScaleMap(outputQuan[0], outputQuan[1]);
    }
    const float* outputPtr = output->readMap<float>();
    float ratio = 0.02;
    if (!checkVector<float>(outputPtr, targetdata.data(), size, ratio)) {
        MNN_ERROR("%s failed: data dimension=[", testName.c_str());
        for (int i = 0; i < dimensions.size(); ++i) {
            if (i < dimensions.size() - 1) {MNN_PRINT("%d, ", dimensions[i]);}
            else {MNN_PRINT("%d], reduce axis=[", dimensions[i]);};
        }
        for (int i = 0; i < reduceAxis.size(); ++i) {
            if (i < reduceAxis.size() - 1) {MNN_PRINT("%d, ", reduceAxis[i]);}
            else {MNN_PRINT("%d]\n", reduceAxis[i]);};
        }
        return false;
    }
    return true;
}

static void computeChannelLayerNorm(const std::vector<float>& input, std::vector<float>& output, int batch, int channel,
                                    const std::vector<float>& gamma, const std::vector<float>& beta,
                                    bool useRMS = false) {
    output.resize(batch * channel);
    for (int n = 0; n < batch; ++n) {
        float mean = 0.0f;
        for (int c = 0; c < channel; ++c) {
            mean += input[n * channel + c];
        }
        mean = useRMS ? 0.0f : mean / channel;
        float variance = 0.0f;
        for (int c = 0; c < channel; ++c) {
            float v = input[n * channel + c] - mean;
            variance += v * v;
        }
        variance /= channel;
        float inv = 1.0f / std::sqrt(variance + eps);
        for (int c = 0; c < channel; ++c) {
            float v = (input[n * channel + c] - mean) * inv;
            if (!gamma.empty()) {
                v = v * gamma[c] + (beta.empty() ? 0.0f : beta[c]);
            }
            output[n * channel + c] = v;
        }
    }
}

static bool checkNC4HW4Logical(VARP output, const std::vector<float>& expected, int batch, int channel,
                               const char* testName) {
    auto logicalOutput = _Convert(output, NCHW);
    auto ptr = logicalOutput->readMap<float>();
    if (!checkVector<float>(ptr, expected.data(), batch * channel, 0.02f)) {
        MNN_ERROR("%s failed!\n", testName);
        return false;
    }
    return true;
}

static VARP makeNC4HW4Input(const std::vector<float>& logical, int batch, int channel, bool rank2 = false) {
    auto input = rank2 ? _Input({batch, channel}, NCHW) : _Input({batch, channel, 1, 1}, NCHW);
    ::memcpy(input->writeMap<float>(), logical.data(), logical.size() * sizeof(float));
    input->unMap();
    return _Convert(input, NC4HW4);
}

class LayerNormTest : public MNNTestCase {
public:
    virtual ~LayerNormTest() = default;
    virtual bool run(int precision) {
        { // test 1.
            std::vector<int32_t> axis = {0, 1, 2};
            std::vector<int> dims = {1, 4, 1, 2};
            // set input data
            std::vector<float> inputdata = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            std::vector<float> tgdata = {0.03527864, 0.0242914, 0.14944272, 0.1714172, -0.24249224, -0.22996021, 0.68665631, 0.66994695};
            std::vector<float> gammaData = {0.1f, 0.2f, 0.3f, 0.4f};
            std::vector<float> betaData = {0.08f, 0.06f, 0.16f, 0.15f};
            
            bool testSuc = testKernel(inputdata, tgdata, dims, axis, eps, gammaData, betaData, {}, {}, "Float LayerNorm Test", 1);
            if (!testSuc) {
                return false;
            }
        }
        { // test 2.
            std::vector<int32_t> axis = {0, 1, 2};
            std::vector<int> dims = {1, 4, 1, 2};
            // set input data
            std::vector<float> inputdata = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            std::vector<float> tgdata = {0.03527864, 0.0242914, 0.14944272, 0.1714172, -0.24249224, -0.22996021, 0.68665631, 0.66994695};
            std::vector<float> inputQuan = {0.063745, 2.0};
            std::vector<float> outputQuan = {0.0095, 0.0};
            std::vector<float> gammaData = {0.1f, 0.2f, 0.3f, 0.4f};
            std::vector<float> betaData = {0.08f, 0.06f, 0.16f, 0.15f};
            
            bool testSuc = testKernel(inputdata, tgdata, dims, axis, eps, gammaData, betaData, inputQuan, outputQuan, "Int8 LayerNorm Test", 1);
            if (!testSuc) {
                return false;
            }
        }
        { // test 3.
            std::vector<int32_t> axis = {0, 1, 2};
            std::vector<int> dims = {1, 4, 1, 2};
            std::vector<float> gammaData = {};
            std::vector<float> betaData = {};
            // set input data
            std::vector<float> inputdata = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            std::vector<float> tgdata = {-0.4472136, -0.55708601, 0.4472136, 0.55708601, -1.34164079, -1.29986737, 1.34164079, 1.29986737};
            
            bool testSuc = testKernel(inputdata, tgdata, dims, axis, eps, gammaData, betaData, {}, {}, "Float LayerNorm Test", 1);
            if (!testSuc) {
                return false;
            }
        }
        { // test 4.
            std::vector<int32_t> axis = {0, 1, 2};
            std::vector<int> dims = {1, 4, 1, 2};
            std::vector<float> gammaData = {};
            std::vector<float> betaData = {};
            // set input data
            std::vector<float> inputdata = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            std::vector<float> tgdata = {-0.4472136, -0.55708601, 0.4472136, 0.55708601, -1.34164079, -1.29986737, 1.34164079, 1.29986737};
            std::vector<float> inputQuan = {0.063745, 2.0};
            std::vector<float> outputQuan = {0.0105, 0.0};
            
            bool testSuc = testKernel(inputdata, tgdata, dims, axis, eps, gammaData, betaData, inputQuan, outputQuan, "Int8 LayerNorm Test", 1);
            if (!testSuc) {
                return false;
            }
        }
        { // test 5.
            std::vector<int32_t> axis = {2, 3};
            std::vector<int> dims = {6, 10, 2, 2};
            std::vector<float> gammaData = {};
            std::vector<float> betaData = {};

            bool testSuc = testKernel(inputdata_0, tgdata_0, dims, axis, eps, gammaData, betaData, {}, {}, "Float LayerNorm Test", 1);
            if (!testSuc) {
                return false;
            }
        }
        {
            std::vector<int32_t> axis = {2, 3};
            std::vector<int> dims = {6, 10, 2, 2};
            std::vector<float> gammaData = {};
            std::vector<float> betaData = {};
            std::vector<float> inputQuan = {0.0752, 0.f};
            std::vector<float> outputQuan = {0.0105, 0.f};
            
            bool testSuc = testKernel(inputdata_0, tgdata_0, dims, axis, eps, gammaData, betaData, inputQuan, outputQuan, "Int8 LayerNorm Test", 1);
            if (!testSuc) {
                return false;
            }
        }
        { // Group Norm without axis 
            std::vector<int> dims = {2, 4, 2, 2};
            auto input = _Input(dims, NCHW);
            bool testSuc = testKernel(inputdata_1, tgdata_1, dims, {}, eps, {}, {}, {}, {}, "Flaot GroupNorm Test", 1, 4);
            if (!testSuc) {
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(LayerNormTest, "op/layernorm");

class LayerNormC4Test : public MNNTestCase {
public:
    virtual ~LayerNormC4Test() = default;
    bool runOne(int batch, int channel, bool useRMS, bool rank2 = false) {
        std::vector<float> logical(batch * channel);
        std::vector<float> gamma(channel);
        std::vector<float> beta(channel);
        for (int i = 0; i < (int)logical.size(); ++i) {
            logical[i] = (float)((i % 17) - 8) * 0.11f + (float)(i % 5) * 0.03f;
        }
        for (int c = 0; c < channel; ++c) {
            gamma[c] = 0.7f + (float)(c % 11) * 0.05f;
            beta[c] = -0.2f + (float)(c % 7) * 0.04f;
        }
        auto input = makeNC4HW4Input(logical, batch, channel, rank2);
        std::unique_ptr<OpT> op(new OpT);
        op->main.type = OpParameter_LayerNorm;
        op->type = OpType_LayerNorm;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        op->main.value = new LayerNormT;
        op->main.AsLayerNorm()->gamma = gamma;
        op->main.AsLayerNorm()->beta = beta;
        op->main.AsLayerNorm()->epsilon = eps;
        op->main.AsLayerNorm()->axis = {rank2 ? -1 : 1};
        op->main.AsLayerNorm()->useRMSNorm = useRMS;
        auto output = Variable::create(Expr::create(std::move(op), {input}));
        std::vector<float> expected;
        computeChannelLayerNorm(logical, expected, batch, channel, gamma, beta, useRMS);
        return checkNC4HW4Logical(output, expected, batch, channel, "LayerNormC4Test");
    }

    virtual bool run(int precision) {
        // batch == 1 takes the flat fast path instead of the packed kernel: at one token
        // the packed offset degenerates to the channel index.
        return runOne(1, 8, false) && runOne(1, 5, false) && runOne(1, 1024, false) && runOne(1, 12, true) &&
               runOne(1, 1024, true) && runOne(1, 1020, true) && runOne(1, 128, true, true) &&
               runOne(2, 8, false) && runOne(3, 5, false) && runOne(13, 1024, false) && runOne(5, 12, false) &&
               runOne(2, 8, true) && runOne(3, 5, true) && runOne(13, 1024, true) && runOne(5, 12, true) &&
               runOne(7, 1020, true) && runOne(22, 128, true, true);
    }
};

class BinaryLayerNormC4Test : public MNNTestCase {
public:
    virtual ~BinaryLayerNormC4Test() = default;
    bool runOne(int batch, int channel, bool useRMS) {
        std::vector<float> logical0(batch * channel);
        std::vector<float> logical1(batch * channel);
        std::vector<float> sumLogical(batch * channel);
        for (int i = 0; i < (int)logical0.size(); ++i) {
            logical0[i] = (float)((i % 17) - 8) * 0.11f + (float)(i % 5) * 0.03f;
            logical1[i] = (float)((i % 13) - 6) * -0.07f + (float)(i % 3) * 0.02f;
        }
        for (int n = 0; n < batch; ++n) {
            for (int c = 0; c < channel; ++c) {
                int logicalIndex = n * channel + c;
                sumLogical[logicalIndex] = logical0[logicalIndex] + logical1[logicalIndex];
            }
        }
        std::vector<float> gamma(channel);
        std::vector<float> beta(channel, 0.0f);
        for (int c = 0; c < channel; ++c) {
            gamma[c] = 0.7f + (float)(c % 11) * 0.05f;
        }
        auto input0 = makeNC4HW4Input(logical0, batch, channel);
        auto input1 = makeNC4HW4Input(logical1, batch, channel);

        std::unique_ptr<OpT> op(new OpT);
        op->main.type = OpParameter_LayerNorm;
        op->type = OpType_LayerNorm;
        op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
        op->main.value = new LayerNormT;
        op->main.AsLayerNorm()->gamma = gamma;
        op->main.AsLayerNorm()->beta = beta;
        op->main.AsLayerNorm()->epsilon = eps;
        op->main.AsLayerNorm()->axis = {1};
        op->main.AsLayerNorm()->useRMSNorm = useRMS;
        auto expr = Expr::create(std::move(op), {input0, input1}, 2);
        auto sumOutput = Variable::create(expr, 0);
        auto normOutput = Variable::create(expr, 1);

        std::vector<float> expectedNorm;
        computeChannelLayerNorm(sumLogical, expectedNorm, batch, channel, gamma, beta, useRMS);
        return checkNC4HW4Logical(sumOutput, sumLogical, batch, channel, "BinaryLayerNormC4SumTest") &&
               checkNC4HW4Logical(normOutput, expectedNorm, batch, channel, "BinaryLayerNormC4NormTest");
    }

    virtual bool run(int precision) {
        // batch == 1 is the decode shape: the fused 2-in/2-out op on a single token,
        // which takes the flat fast path rather than the packed kernel. The last case
        // has batch >= 256, which is where Metal switches to the row-parallel prefill
        // kernel; the small batches never reach it.
        return runOne(1, 8, false) && runOne(1, 5, false) && runOne(1, 1024, false) && runOne(1, 12, true) &&
               runOne(1, 1024, true) && runOne(1, 1020, true) &&
               runOne(2, 8, false) && runOne(3, 5, false) && runOne(13, 1024, false) && runOne(5, 12, false) &&
               runOne(2, 8, true) && runOne(3, 5, true) && runOne(13, 1024, true) && runOne(5, 12, true) &&
               runOne(7, 1020, true) && runOne(300, 512, true);
    }
};

MNNTestSuiteRegister(LayerNormC4Test, "op/layernorm/c4");
MNNTestSuiteRegister(BinaryLayerNormC4Test, "op/layernorm/c4_binary");

// The fused residual output must follow the same finite-range conversion as
// the flat path. Normalization must still use the unrounded FP32 sum.
class LayerNormFp16ResidualSaturationTest : public MNNTestCase {
public:
    bool run(int precision) override {
        if (precision != BackendConfig::Precision_Low && precision != BackendConfig::Precision_Normal) {
            return true;
        }
        int cases = 0, failures = 0;
        bool checkedBackend = false;
        for (int batch : {1, 2, 5, 13}) {
            for (int channel : {8, 1, 3, 5, 9, 16, 17, 128}) {
                for (bool rms : {false, true}) {
                    for (bool affine : {false, true}) {
                        for (int pattern = 0; pattern < 6; ++pattern) {
                            std::vector<float> a(batch * channel), b(a.size()), exact(a.size()), stored(a.size());
                            std::vector<float> gamma, beta;
                            if (affine) {
                                gamma.resize(channel);
                                beta.resize(channel);
                                for (int c = 0; c < channel; ++c) {
                                    gamma[c] = 0.75f + (c % 3) * 0.125f;
                                    beta[c] = (c % 3 - 1) * 0.125f;
                                }
                            }
                            for (int n = 0; n < batch; ++n) {
                                for (int c = 0; c < channel; ++c) {
                                    const int i = n * channel + c;
                                    if (pattern < 2) {
                                        a[i] = b[i] = pattern == 0 ? 65504.0f : -65504.0f;
                                    } else if (pattern == 2) {
                                        a[i] = b[i] = (c % 2 == 0 ? 1.0f : -1.0f) * (c % 3 == 0 ? 65504.0f : 32752.0f);
                                    } else if (pattern == 3) {
                                        a[i] = c % 2 == 0 ? 65504.0f : -65504.0f;
                                        b[i] = (c % 3 - 1) * 32.0f;
                                    } else if (pattern == 4) {
                                        a[i] = (c % 9 - 4) * 0.25f;
                                        b[i] = (c % 5 - 2) * 0.125f;
                                    } else {
                                        // Clipping before statistics would change the relative magnitudes.
                                        a[i] = c % 2 == 0 ? 65504.0f : 32752.0f;
                                        b[i] = c % 2 == 0 ? 65504.0f : 0.0f;
                                    }
                                    exact[i] = a[i] + b[i];
                                    stored[i] = precision == BackendConfig::Precision_Low
                                                    ? std::min(std::max(exact[i], -65504.0f), 65504.0f)
                                                    : exact[i];
                                }
                            }
                            auto makeNorm = [&](const std::vector<VARP>& inputs, bool useRMS,
                                                const std::vector<float>& g, const std::vector<float>& bias) {
                                std::unique_ptr<OpT> op(new OpT);
                                op->type = OpType_LayerNorm;
                                op->main.type = OpParameter_LayerNorm;
                                op->main.value = new LayerNormT;
                                op->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
                                auto param = op->main.AsLayerNorm();
                                param->axis = {1};
                                param->epsilon = eps;
                                param->useRMSNorm = useRMS;
                                param->gamma = g;
                                param->beta = bias;
                                return Expr::create(std::move(op), inputs, inputs.size() == 2 ? 2 : 1);
                            };
                            auto expr =
                                makeNorm({makeNC4HW4Input(a, batch, channel), makeNC4HW4Input(b, batch, channel)}, rms,
                                         gamma, beta);
                            auto sum = Variable::create(expr, 0);
                            auto norm = Variable::create(expr, 1);
                            Variable::prepareCompute({sum, norm});
                            if (sum->readMap<float>() == nullptr) {
                                return false;
                            }
                            if (!checkedBackend) {
                                auto backend = TensorUtils::getDescribeOrigin(sum->getTensor())->getBackend();
                                auto expectedBackend = precision == BackendConfig::Precision_Low
                                                           ? MNN_FORWARD_CPU_EXTENSION
                                                           : MNN_FORWARD_CPU;
                                if (backend == nullptr || backend->type() != expectedBackend) {
                                    MNN_PRINT("Residual saturation test: requested native CPU precision unavailable\n");
                                    return true;
                                }
                                checkedBackend = true;
                            }
                            auto sumHost = _Convert(sum, NCHW);
                            auto normHost = _Convert(norm, NCHW);
                            // This consumer exposes Inf propagation into a subsequent normalization.
                            auto next = Variable::create(makeNorm({sum}, true, {}, {}));
                            auto nextHost = _Convert(next, NCHW);
                            const auto* sumPtr = sumHost->readMap<float>();
                            const auto* normPtr = normHost->readMap<float>();
                            const auto* nextPtr = nextHost->readMap<float>();
                            if (!sumPtr || !normPtr || !nextPtr) {
                                return false;
                            }
                            std::vector<float> expectedNorm, expectedNext;
                            computeChannelLayerNorm(exact, expectedNorm, batch, channel, gamma, beta, rms);
                            computeChannelLayerNorm(stored, expectedNext, batch, channel, {}, {}, true);
                            bool valid = true;
                            for (int i = 0; i < batch * channel; ++i) {
                                if (!std::isfinite(sumPtr[i]) || sumPtr[i] != stored[i] || !std::isfinite(normPtr[i]) ||
                                    std::fabs(normPtr[i] - expectedNorm[i]) >
                                        0.003f * (1.0f + std::fabs(expectedNorm[i])) ||
                                    !std::isfinite(nextPtr[i]) ||
                                    std::fabs(nextPtr[i] - expectedNext[i]) >
                                        0.003f * (1.0f + std::fabs(expectedNext[i]))) {
                                    if (failures < 4) {
                                        MNN_ERROR(
                                            "Residual saturation mismatch: batch=%d channel=%d rms=%d affine=%d "
                                            "pattern=%d index=%d expectedSum=%g sum=%g norm=%g next=%g\n",
                                            batch, channel, rms, affine, pattern, i, stored[i], sumPtr[i], normPtr[i],
                                            nextPtr[i]);
                                    }
                                    valid = false;
                                    break;
                                }
                            }
                            failures += !valid;
                            ++cases;
                        }
                    }
                }
            }
        }
        MNN_PRINT("Residual saturation: %d cases, %d failures\n", cases, failures);
        return failures == 0;
    }
};
MNNTestSuiteRegister(LayerNormFp16ResidualSaturationTest, "op/layernorm/fp16_residual_saturation");
