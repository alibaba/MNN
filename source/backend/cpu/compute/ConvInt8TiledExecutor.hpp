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

class ConvInt8TiledExecutor : public CPUConvolution {
public:
    // given weight+bias+scale, do post process
    ConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, std::shared_ptr<ResourceInt8> res);
    virtual ~ConvInt8TiledExecutor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    virtual void getPackParameter(int* Unit, int* SrcUnit, int* DestUnit, const CoreInt8Functions* core) = 0;
    static void reorderWeight(Tensor* weight, const uint8_t* weightSrc, int SRC_UNIT, int UNIT, int ic, int oc, int kernelCount, int pack);

protected:
    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    int mTileCount;
    int mThreadNums;
    std::shared_ptr<Tensor> mTempIm2ColBuffer;
    std::shared_ptr<CPUConvolution::ResourceInt8> mResourceInt8;
    // std::shared_ptr<CPUConvolution::Resource> mResource;
    CPUConvolution::MutableResourceInt8 mMutableResource;
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
    // given weight+bias+scale, do post process
    DenseConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, std::shared_ptr<ResourceInt8> res, bool dynamicQuantExe);
    virtual ~DenseConvInt8TiledExecutor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void getPackParameter(int* Unit, int* SrcUnit, int* DestUnit, const CoreInt8Functions* core) override;
private:
    DenseConvInt8TiledExecutor(Backend* backend, const Convolution2D* common, bool dynamicQuantExe, const DenseConvInt8TiledExecutor& exe);

    decltype(CoreInt8Functions::Int8GemmKernel) mGemmKernel;
    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, ssize_t)> mQuantFunc;
    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, ssize_t, size_t, size_t)> mQuantAndReorderFunc = nullptr;
    std::function<void(float* dest, int8_t* source, const float* scale, ssize_t realDstCount, SumByAxisParams sumParams)> mSumByAxisLFunc;
    std::shared_ptr<Tensor> mQuantInput;
    std::shared_ptr<Tensor> mDynamicBias;
    std::shared_ptr<Tensor> mScaleFuse;
    std::shared_ptr<Tensor> mBatchQuantInfo;
    std::shared_ptr<Tensor> mInputDeqScales;
    std::shared_ptr<Tensor> mTempMaxMinValueBuffer;
    std::shared_ptr<CPUConvolution::Resource> mResource;
    std::vector<uint8_t> mTempSrcSum;
    std::vector<int32_t> mDivides;

    int mThreadNums;
    int mBlockNum;
    int mOcPerThread;
    bool mDynamicQuantExe;
    bool mSplitByOc;
    bool mUseBatchQuan;
};

} // namespace MNN

#endif /* ConvInt8TiledExecutor_hpp */
