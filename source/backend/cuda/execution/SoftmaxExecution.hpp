//
//  SoftmaxExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SoftmaxExecution_hpp
#define SoftmaxExecution_hpp

#include "core/Execution.hpp"

#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

class SoftmaxExecution : public Execution {
public:
    SoftmaxExecution(int axis, Backend *backend);
    virtual ~SoftmaxExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnDataType_t cudnn_data_type_;
    
    int mAxis;
    int axis;
    int inside;
    int outside;
};

} // namespace CUDA
} // namespace MNN
#endif /* SoftmaxExecution_hpp */