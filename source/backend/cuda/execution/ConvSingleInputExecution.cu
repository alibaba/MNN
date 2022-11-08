//
//  ConvSingleInputExecution.cpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvSingleInputExecution.hpp"
#include "ConvWinogradExecution.hpp"
#include "ConvCutlassExecution.hpp"
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

#ifdef USE_MNN_CONV

        std::shared_ptr<ConvSingleInputExecution::Resource> resource(new ConvSingleInputExecution::Resource(backend, op));
        return new ConvSingleInputExecution(backend, op, resource);

#else
        auto conv = op->main_as_Convolution2D()->common();
        if(ConvWinogradExecution::isValid(op->main_as_Convolution2D())) { // inputs[0] is invalid now.
            //printf("%dx%ds%dd%d\n", conv->kernelX(), conv->kernelY(), conv->strideX(), conv->dilateX());

            std::shared_ptr<ConvWinogradExecution::Resource> resource(new ConvWinogradExecution::Resource(backend, op));
            return new ConvWinogradExecution(backend, op, resource);
        }

        std::shared_ptr<ConvCutlassExecution::Resource> resource(new ConvCutlassExecution::Resource(backend, op));
        return new ConvCutlassExecution(backend, op, resource);
#endif

    }
};

CUDACreatorRegister<CUDAConvolutionCreator> __ConvExecution(OpType_Convolution);

}// namespace CUDA
}// namespace MNN
