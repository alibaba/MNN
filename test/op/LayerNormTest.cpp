//
//  LayerNormTest.cpp
//  MNNTests
//
//  Created by MNN on 2023/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include <cmath>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

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
            if (i < dimensions.size() - 1) MNN_PRINT("%d, ", dimensions[i]);
            else MNN_PRINT("%d], reduce axis=[", dimensions[i]);
        }
        for (int i = 0; i < reduceAxis.size(); ++i) {
            if (i < reduceAxis.size() - 1) MNN_PRINT("%d, ", reduceAxis[i]);
            else MNN_PRINT("%d]\n", reduceAxis[i]);
        }
        return false;
    }
    return true;
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
