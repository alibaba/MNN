//
//  ConvSingleInputExecution.cpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvSingleInputExecution.hpp"
#include "ConvWinogradExecution.hpp"
#include "ConvImplicitExecution.hpp"
#include "ConvCutlassExecution.hpp"
#include "MultiInputConvExecution.hpp"
#ifdef ENABLE_CUDA_QUANT
#include "int8/ConvInt8CutlassExecution.hpp"
#endif
#ifdef MNN_LOW_MEMORY
#include "weight_only_quant/ConvFpAIntBExecution.hpp"
#endif
#include "bf16/ConvCutlassBf16Execution.hpp"
#include "backend/cuda/core/CUDATools.hpp"

namespace MNN {
namespace CUDA {

class CUDAConvolutionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
            const MNN::Op* op, Backend* backend) const override {
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                if (quan->has_scaleInt()) {
                    // Don't support IDST-int8 because of error
                    return nullptr;
                }
            }
        }

        #ifdef MNN_LOW_MEMORY
        auto conv2dParams = op->main_as_Convolution2D();
        bool isMemoryLowWeightOnlyQuant = (conv2dParams->quanParameter() != nullptr && conv2dParams->quanParameter()->buffer() != nullptr);
        isMemoryLowWeightOnlyQuant = isMemoryLowWeightOnlyQuant && (static_cast<CUDABackend*>(backend)->getMemoryMode() == BackendConfig::Memory_Low);
        isMemoryLowWeightOnlyQuant = isMemoryLowWeightOnlyQuant && ConvFpAIntBExecution::isValid(op->main_as_Convolution2D(), backend);
        if (isMemoryLowWeightOnlyQuant) {
            std::shared_ptr<ConvFpAIntBExecution::Resource> resource(new ConvFpAIntBExecution::Resource(backend, op));
            return new ConvFpAIntBExecution(backend, op, resource);
        }
        #endif

        if (inputs.size() == 2 || inputs.size() == 3) {
            return new MultiInputConvExecution(op, backend);
        }

        auto conv = op->main_as_Convolution2D()->common();
        if(ConvImplicitExecution::isValid(op->main_as_Convolution2D(), inputs[0], outputs[0], backend)) { // inputs[0] is invalid now.
            std::shared_ptr<ConvImplicitExecution::Resource> resource(new ConvImplicitExecution::Resource(backend, op));
            return new ConvImplicitExecution(backend, op, resource);
        }
        if(ConvWinogradExecution::isValid(op->main_as_Convolution2D())) { // inputs[0] is invalid now.
            //printf("%dx%ds%dd%d\n", conv->kernelX(), conv->kernelY(), conv->strideX(), conv->dilateX());

            std::shared_ptr<ConvWinogradExecution::Resource> resource(new ConvWinogradExecution::Resource(backend, op));
            return new ConvWinogradExecution(backend, op, resource);
        }

        #ifdef ENABLE_CUDA_BF16
        if (static_cast<CUDABackend*>(backend)->getPrecision() == 3) {
            std::shared_ptr<ConvCutlassBf16Execution::Resource> resource(new ConvCutlassBf16Execution::Resource(backend, op));
            return new ConvCutlassBf16Execution(backend, op, resource);
        }
        #endif

        std::shared_ptr<ConvCutlassExecution::Resource> resource(new ConvCutlassExecution::Resource(backend, op));
        return new ConvCutlassExecution(backend, op, resource);
    }
};

#ifdef ENABLE_CUDA_QUANT
class CUDAConvolutionInt8Creator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
            const MNN::Op* op, Backend* backend) const override {
        std::shared_ptr<ConvInt8CutlassExecution::Resource> resource(new ConvInt8CutlassExecution::Resource(backend, op));
        return new ConvInt8CutlassExecution(backend, op, resource);
    }
};

CUDACreatorRegister<CUDAConvolutionInt8Creator> __ConvInt8Execution(OpType_ConvInt8);
#endif

CUDACreatorRegister<CUDAConvolutionCreator> __ConvExecution(OpType_Convolution);

}// namespace CUDA
}// namespace MNN
