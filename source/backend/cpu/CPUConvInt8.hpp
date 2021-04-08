//
//  CPUConvInt8.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConvInt8_hpp
#define CPUConvInt8_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "compute/Int8FunctionsOpt.h"

namespace MNN {

class CPUConvInt8 : public CPUConvolution {
public:
    struct ResourceInt8 {
        std::shared_ptr<Tensor> mWeightInt8;
        std::shared_ptr<Tensor> mBiasInt32;
        std::shared_ptr<Tensor> mScaleFloat;
        void (*mGemmKernel)(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                            size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDstCount);
        // relu or relu6
        bool mRelu;
        int mActBits;

        int8_t mInputZeroPoint;
        int8_t mOutputZeroPoint;
        int8_t mClampMin;
        int8_t mClampMax;
        Backend* backend;
        float mInputScale;
        float mOutputScale;
#ifdef MNN_USE_SSE
        std::vector<int> offsets;
#endif
        void updateInputOutputScale(float inputScale, float outputScale);
        ~ ResourceInt8();
    };
    CPUConvInt8(Backend *backend, const Convolution2DCommon* common, std::shared_ptr<ResourceInt8> resource);
    virtual ~CPUConvInt8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    static std::shared_ptr<ResourceInt8> makeResource(Backend *backend, const MNN::Convolution2D *convOp,
                                                      float inputScale, float outputScale);
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    std::shared_ptr<ResourceInt8> mResource;
    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    int mTileCount;
    int mThreadNums;
    float mInputScale;
    float mOutputScale;
    Tensor mTempIm2ColBuffer;
};

} // namespace MNN

#endif /* CPUConvInt8_hpp */
