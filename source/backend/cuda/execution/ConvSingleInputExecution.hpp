//
//  ConvSingleInputExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvSingleInputExecution_hpp
#define ConvSingleInputExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"

namespace MNN {
namespace CUDA {

struct KernelInfo {
    int groups         = 0;
    int kernelN        = 0;
    int kernelC        = 0;
    int kernelX        = 0;
    int kernelY        = 0;
    int strideX        = 0;
    int strideY        = 0;
    int dilateX        = 0;
    int dilateY        = 0;
    int activationType = 0;
};//

} // namespace CUDA
} // namespace MNN

#endif /* ConvSingleInputExecution */
