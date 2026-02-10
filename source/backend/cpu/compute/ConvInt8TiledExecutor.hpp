//
//  ConvInt8TiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvInt8TiledExecutor_hpp
#define ConvInt8TiledExecutor_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "Int8FunctionsOpt.h"
#include "CommonOptFunction.h"

namespace MNN {
typedef void (*weightSummerFuncion)(float* kernelsum, int8_t* source, size_t outside, size_t reduceAxis, size_t hP, size_t lP);
class ConvInt8TiledExecutor : public CPUConvolution {
public:
    // given weight+bias+scale, do post process
    ConvInt8TiledExecutor(Backend* backend, const Op* op);
    ConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ResourceInt8> res);
    virtual ~ConvInt8TiledExecutor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    static void packWeightAndQuantInfo(int8_t* dstbuffer, const int8_t* weight, const int8_t* quantInfo, int32_t* info, int infoBytes = 4);
    static void reorderWeight(uint8_t* dst, const uint8_t* src, int32_t* info, int32_t initval = 0, float* kernelsum = nullptr, weightSummerFuncion summerFunc = nullptr);
    static void initializeConvInt8QuantInfo(std::shared_ptr<CPUConvolution::ResourceInt8>& resourceInt8, const Convolution2D* conv2D, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon);

protected:
    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    int mTileCount;
    int mThreadNums;
    std::shared_ptr<Tensor> mTempIm2ColBuffer;
    std::shared_ptr<CPUConvolution::ResourceInt8> mResourceInt8;
    std::shared_ptr<CPUConvolution::MutableResourceInt8> mMutableResource;
    MemChunk mBlitInfo;
    std::pair<size_t, size_t> mBlitInfoStride;
    int mIm2ColCount;
};

//
//  DenseConvInt8TiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//


class DenseConvInt8TiledExecutor : public ConvInt8TiledExecutor {
public:
    DenseConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant); // dynamic quant
    virtual ~DenseConvInt8TiledExecutor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    DenseConvInt8TiledExecutor(Backend* backend, const Op* op, const DenseConvInt8TiledExecutor& exe);

    decltype(CoreInt8Functions::Int8GemmKernel) mGemmKernel;
    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, const float*, ssize_t)> mQuantFunc;
    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, const float*, size_t, size_t)> mQuantAndReorderFunc = nullptr;
    std::function<void(float* dest, int8_t* source, const float* scale, ssize_t realDstCount, SumByAxisParams sumParams)> mSumByAxisLFunc;
    std::shared_ptr<Tensor> mQuantInput;
    std::shared_ptr<Tensor> mDynamicBias;
    std::shared_ptr<Tensor> mAccumBuffer;
    std::shared_ptr<Tensor> mBatchQuantInfo;
    MemChunk mTempMaxMinValueBuffer;
    MemChunk mTempSrcSum;
    MemChunk mQScaleZero;
    MemChunk mReorderBuffer;
    MemChunk mBiasBufferFusedInputzero;
    MemChunk mWeight4Prefill;
    MemChunk mWeightKernelSum4Prefill;
    // for 4Bit Ptq model
    MemChunk mTempOutput;
    std::vector<int32_t> mDivides;
    std::vector<int32_t> mDividesTmp;
    std::vector<decltype(CoreInt8Functions::Int8GemmKernel)> mGemmKernels;

    int mGemmUnits[3];
    int mThreadNums;
    int mBlockNum = 1;
    int mInputBlockNum = 1;
    int mOcPerThread;
    int mOcMain;
    int mOcBranch = 0;
    int mRatioPrefill;
    int mRatioDecode;
    int mSmeCores = 2;
    int mOriginSmeWork = 0;
    int mSizeInputBlockQuant;
    bool mSplitByOc;
    bool mUseBatchQuan;
    bool mIm2ColBasedInt8;
    bool mToFuseInputbias2Bias;
    bool mOnlineReorderWeightSme = false;

    // for 4Bit Ptq model
    bool m4BitPtq = false;
    bool mMixedKernel;
    MatmulRelatedFunctions mRelatedFunctions;
    MatmulRelatedFunctions mArm82Functions;
};

} // namespace MNN

#endif /* ConvInt8TiledExecutor_hpp */
