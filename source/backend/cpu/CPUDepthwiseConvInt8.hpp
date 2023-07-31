//
//  CPUDepthwiseConvInt8.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDepthwiseConvInt8_hpp
#define CPUDepthwiseConvInt8_hpp

#include "CPUConvolution.hpp"
#include "compute/Int8FunctionsOpt.h"
namespace MNN {

class CPUDepthwiseConvInt8 : public CPUConvolution {
public:
    CPUDepthwiseConvInt8(Backend *backend, const Convolution2DCommon* common, std::shared_ptr<ResourceInt8> res);
    virtual ~CPUDepthwiseConvInt8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    int mThreadNumber;
    int mPack = 16;
    bool mUse3x3Kernel = false;
    std::shared_ptr<CPUConvolution::ResourceInt8> mResource;
    std::shared_ptr<Tensor> mInputPad;
    std::pair<int, int> mPads;
    std::pair<int, int> mPaddedSize;
    std::pair<int, int> mStrides;
    std::pair<int, int> mDilates;
    std::pair<int, int> mKernels;
    MutableResourceInt8 mMutableResource;
    std::shared_ptr<Tensor> mInputTemp;
    std::shared_ptr<Tensor> mOutputTemp;
    std::shared_ptr<Tensor> mWeightTemp;
    void fastDepthwiseInt8(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    std::function<void(int8_t* dst, const int8_t* src, const int8_t* weight, const QuanPostTreatParameters* parameters, size_t width,
                       size_t src_w_step, size_t fw, size_t fh, size_t dilateX_step, size_t dilateY_step, int8_t* idxOrder)> mThreadFunction;
    std::vector<int8_t> mOrder;
    std::vector<int32_t> mBiasExtend;
};

} // namespace MNN

#endif /* CPUDepthwiseConvInt8_hpp */
