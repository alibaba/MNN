//
//  DeconvSingleInputExecution.hpp
//  MNN
//
//  Created by MNN on 2020/08/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DeconvSingleInputExecution_hpp
#define DeconvSingleInputExecution_hpp

#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "TensorCoreGemm.cuh"

namespace MNN {
namespace CUDA {

struct IOInfo {
    int ib, ic, ih, iw;
    int ob, oc, oh, ow;
};
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

struct Col2ImParameter {
    int padX;
    int padY;
    int dilateX;
    int dilateY;
    int strideX;
    int strideY;
    int kernelX;
    int kernelY;
    int oc;
    int ic;
    int iw;
    int ih;
    int ow;
    int oh;
    int ob;
};

struct InputReorderParameter {
    int ic_stride;
    int ib_stride;
    int oc_stride;
    int ob_stride;
    int hw_size;
    int l_size;
    int h_size;
    int lpack_size;
    int hpack_size;
}; 


extern "C"
class DeconvSingleInputExecution : public Execution {
public:
    struct Resource {
        Resource(Backend* bn, const MNN::Op* op);
        ~ Resource();
        void* mFilter;
        void* mBias;
        std::shared_ptr<Tensor> weightTensor;
        std::shared_ptr<Tensor> biasTensor;
        KernelInfo mKernelInfo;
        Backend* mBackend = nullptr;
    };
    DeconvSingleInputExecution(Backend* backend, const MNN::Op* op,  std::shared_ptr<Resource> res);
    virtual ~DeconvSingleInputExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<Resource> mResource;

    const Op* mOp = nullptr;
    MatMulParam mMatMulParam;
    std::pair<void*, int> mGpuMatMulParam;

    Col2ImParameter mCol2ImParamter;
    std::pair<void*, int> mGpuCol2ImParam;

    InputReorderParameter mInpReorderParameter;
    std::pair<void*, int> mGpuInpReorderParam;

    float* mIm2ColBuffer;
    __half* mInputBuffer;
};

} // namespace CUDA
} // namespace MNN

#endif /* DeconvSingleInputExecution */
