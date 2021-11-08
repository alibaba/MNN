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
#include "ConvInt8Winograd.hpp"
#include "Int8FunctionsOpt.h"

namespace MNN {

class ConvInt8TiledExecutor : public CPUConvolution {
public:
    // given weight+bias+scale, do post process
    ConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, std::shared_ptr<ResourceInt8> res);
    // only given weight, not do post process
    ConvInt8TiledExecutor(Backend* backend, const Convolution2DCommon* common, std::shared_ptr<Tensor> weight, bool fastgemm);
    virtual ~ConvInt8TiledExecutor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    virtual void getPackParameter(int* Unit, int* SrcUnit, int* DestUnit, const CoreInt8Functions* core) = 0;
protected:
    ConvInt8TiledExecutor(Backend* backend, const Convolution2DCommon* common, const ConvInt8TiledExecutor& exe);
    friend class ConvInt8Winograd;

protected:
    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    int mTileCount;
    int mThreadNums;
    std::shared_ptr<Tensor> mTempIm2ColBuffer;
    bool mDoPostProcess = true; //whether quan post process (add bias, min/max then scale to int8)
    std::shared_ptr<CPUConvolution::ResourceInt8> mResource;

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
    DenseConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, std::shared_ptr<ResourceInt8> res);
    // only given weight, not do post process
    DenseConvInt8TiledExecutor(Backend* backend, const Convolution2DCommon* common, std::shared_ptr<Tensor> weight, bool fastgemm);
    virtual ~DenseConvInt8TiledExecutor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void getPackParameter(int* Unit, int* SrcUnit, int* DestUnit, const CoreInt8Functions* core) override;
private:
    DenseConvInt8TiledExecutor(Backend* backend, const Convolution2DCommon* common, const DenseConvInt8TiledExecutor& exe);
    friend class ConvInt8Winograd;

    decltype(CoreInt8Functions::Int8GemmKernel) mGemmKernel;

};

} // namespace MNN

#endif /* ConvInt8TiledExecutor_hpp */
