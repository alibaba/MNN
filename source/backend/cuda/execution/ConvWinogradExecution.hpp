//
//  ConvWinogradExecution.hpp
//  MNN
//
//  Created by MNN on 2022/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvWinogradExecution_hpp_
#define ConvWinogradExecution_hpp_

#include "ConvSingleInputExecution.hpp"
#include "TensorCoreGemmPacked.cuh"

namespace MNN {
namespace CUDA {

class ConvWinogradExecution : public Execution {
public:
    struct Resource;
    static bool isValid(const Convolution2D* conv, const Tensor* input);
    ConvWinogradExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res);
    virtual ~ConvWinogradExecution();

    struct Resource {
        Resource(Backend* backend, const MNN::Op* op);
        ~ Resource();
        void* mFilter;
        void* mBias;
        std::shared_ptr<Tensor> weightTensor;
        std::shared_ptr<Tensor> biasTensor;
        KernelInfo mKernelInfo;
        Backend* mBackend = nullptr;
        bool mUseHPack = false;
    };

    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    
private:
    std::shared_ptr<Resource> mResource;
    const Op* mOp = nullptr;
    __half* mBtdB_Buffer;
    void* mMatmul_Buffer;
    MatMulParam mMatMulParam;
    std::pair<void*, int> mGpuMatMulParam;

    int mPadX;
    int mPadY;
    int mBlock2;
};

} // namespace CUDA
} // namespace MNN
#endif /* ConvWinogradExecution_hpp_ */