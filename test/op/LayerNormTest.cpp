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

VARP _LayerNorm(VARP x, std::vector<int32_t> axis, float epsilon, std::vector<float> gamma, std::vector<float> beta) {
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
    return (Variable::create(Expr::create(std::move(op), {x})));
}

static void _refLayerNorm(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t in_size, size_t out_size) {
    for(int i = 0; i < out_size; ++i){
        const float* input = src + i * in_size;
        float* output = dst + i * in_size;
        float sum = 0.f;
        for (int j = 0; j < in_size; ++j) {
            sum += input[j];
        }
        float mean = sum / in_size;
        float square_sum = 0.f;
        for (int j = 0; j < in_size; ++j) {
            square_sum += (input[j] - mean) * (input[j] - mean);
        }
        float variable = square_sum / in_size;
        variable = 1.f / std::sqrt(variable + epsilon);
        
        if (gamma && beta) {
            for (int j = 0; j < in_size; ++j) {
                output[j] = (input[j] - mean) * variable * gamma[j] + beta[j];
            }
        } else {
            for (int j = 0; j < in_size; ++j) {
                output[j] = (input[j] - mean) * variable;
            }
        }
    }
}
class LayerNormTest : public MNNTestCase {
public:
    virtual ~LayerNormTest() = default;
    virtual bool run(int precision) {
        {
            std::vector<int> dims = {1, 4, 1, 2};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<float> gammaData = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<float> betaData = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<int32_t> axis = {0, 1, 2};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, gammaData, betaData);
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, gammaData.data(), betaData.data(), eps, inner_size, outter_size);
            auto gotOutput                        = output->readMap<float>();
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 1000;
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 1e-5 * errorScale)) {
                MNN_ERROR("Float LayerNormTest axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        // LayerNorm Int8 test.
        {
            std::vector<int> dims = {1, 4, 1, 2};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            input->writeScaleMap(0.063745, 2.0);
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<float> gammaData = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<float> betaData = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<int32_t> axis = {0, 1, 2};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, gammaData, betaData);
            output->writeScaleMap(0.0095, 0.0);
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, gammaData.data(), betaData.data(), eps, inner_size, outter_size);
            auto gotOutput                        = output->readMap<float>();
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 1000;
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
                MNN_ERROR("Int8 LayerNormTest axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        {
            std::vector<int> dims = {1, 4, 1, 2};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<int32_t> axis = {0, 1, 2};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, {}, {});
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, nullptr, nullptr, eps, inner_size, outter_size);
            auto gotOutput                        = output->readMap<float>();
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 1000;
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 1e-5 * errorScale)) {
                MNN_ERROR("Float LayerNormTest without gamma beta axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        {
            std::vector<int> dims = {1, 4, 1, 2};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            input->writeScaleMap(0.063745, 2.0);
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<int32_t> axis = {0, 1, 2};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, {}, {});
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, nullptr, nullptr, eps, inner_size, outter_size);
            output->writeScaleMap(0.01087, 0.0);
            auto gotOutput                        = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
                MNN_ERROR("Int8 LayerNormTest without gamma beta axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        {
            std::vector<int> dims = {1, 2, 2, 2};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<float> gammaData = {0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<float> betaData = {0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<int32_t> axis = {0, 1};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, gammaData, betaData);
            std::vector<float> expectedOutput(8);
            
            int axis_size = axis.size();
            int rank = dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, gammaData.data(), betaData.data(), eps, inner_size, outter_size);
            auto gotOutput                        = output->readMap<float>();
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 1000;
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 1e-5 * errorScale)) {
                MNN_ERROR("Float LayerNormTest axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        {
            std::vector<int> dims = {1, 2, 2, 2};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            input->writeScaleMap(0.063745, 2.0);
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<float> gammaData = {0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<float> betaData = {0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<int32_t> axis = {0, 1};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, gammaData, betaData);
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, gammaData.data(), betaData.data(), eps, inner_size, outter_size);
            output->writeScaleMap(0.0084, 0.0);
            auto gotOutput                        = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
                MNN_ERROR("Int8 LayerNormTest axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        {
            std::vector<int> dims = {1, 2, 2, 2};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<int32_t> axis = {0, 1};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, {}, {});
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, nullptr, nullptr, eps, inner_size, outter_size);
            auto gotOutput                        = output->readMap<float>();
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 1000;
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 1e-5 * errorScale)) {
                MNN_ERROR("Float LayerNormTest without gamma beta axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        {
            std::vector<int> dims = {1, 2, 2, 2};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            input->writeScaleMap(0.064, 2.0);
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<int32_t> axis = {0, 1};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, {}, {});
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, nullptr, nullptr, eps, inner_size, outter_size);
            output->writeScaleMap(0.0089, 0.0);
            auto gotOutput                        = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
                MNN_ERROR("Int8 LayerNormTest without gamma beta axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        {
            std::vector<int> dims = {1, 2, 1, 4};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<float> gammaData = {0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<float> betaData = {0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<int32_t> axis = {0};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, gammaData, betaData);
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, gammaData.data(), betaData.data(), eps, inner_size, outter_size);
            auto gotOutput                        = output->readMap<float>();
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 1000;
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 1e-5 * errorScale)) {
                MNN_ERROR("Float LayerNormTest axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        {
            std::vector<int> dims = {1, 2, 1, 4};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            input->writeScaleMap(0.064, 2.0);
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<float> gammaData = {0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<float> betaData = {0.5f, 0.5f, 0.5f, 0.5f};
            std::vector<int32_t> axis = {0};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, gammaData, betaData);
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, gammaData.data(), betaData.data(), eps, inner_size, outter_size);
            output->writeScaleMap(0.0099, 0.0);
            auto gotOutput                        = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
                MNN_ERROR("Int8 LayerNormTest axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        {
            std::vector<int> dims = {1, 2, 1, 4};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<int32_t> axis = {0};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, {}, {});
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, nullptr,nullptr,  eps, inner_size, outter_size);
            auto gotOutput                        = output->readMap<float>();
            float errorScale = precision <= MNN::BackendConfig::Precision_High ? 1 : 1000;
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 1e-5 * errorScale)) {
                MNN_ERROR("Float LayerNormTest without gamma beta axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        {
            std::vector<int> dims = {1, 2, 1, 4};
            auto input = _Input(dims, NCHW);
            // set input data
            const float inpudata[] = {-1.0, -2.0, 3.0, 4.0, -5.0, -6.0, 7.0, 8.0};
            input->writeScaleMap(0.06375, 2.0);
            auto inputPtr          = input->writeMap<float>();
            memcpy(inputPtr, inpudata, 8 * sizeof(float));
            std::vector<int32_t> axis = {0};
            float eps = 1.00f;
            auto output = _LayerNorm(input, axis, eps, {}, {});
            std::vector<float> expectedOutput(8);
            
            int axis_size = (int)axis.size();
            int rank = (int)dims.size();
            int outter_size = 1;
            int inner_size = 1;
            for (int i = 0; i < rank - axis_size; ++i) {
                outter_size *= dims[i];
            }
            for (int i = rank - axis_size; i < rank; ++i) {
                inner_size *= dims[i];
            }
            _refLayerNorm(expectedOutput.data(), inpudata, nullptr,nullptr,  eps, inner_size, outter_size);
            output->writeScaleMap(0.00858, 0.f);
            auto gotOutput                        = output->readMap<float>();
            if (!checkVectorByRelativeError<float>(gotOutput, expectedOutput.data(), 8, 0.01)) {
                MNN_ERROR("Int8 LayerNormTest without gamma beta axis = %d test failed!\n", axis_size);
                return false;
            }
        }
        return true;
    }
};
MNNTestSuiteRegister(LayerNormTest, "op/layernorm");
