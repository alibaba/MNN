//
//  ConvInt8Winograd.hpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvInt8Winograd_hpp
#define ConvInt8Winograd_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "backend/cpu/compute/Int8FunctionsOpt.h"

namespace MNN {
class ConvInt8Winograd : public CPUConvolution {
public:
    using CommonPair = std::pair<const Convolution2DCommon*, unsigned char*>;
    struct UnitAttr {
        int kyStart;
        int kySize;
        int kxStart;
        int kxSize;
        int unitY;
        int unitX;
    };
    struct Unit {
        UnitAttr attr;
        std::shared_ptr<CommonPair> common;
        std::shared_ptr<Tensor> input;
        std::shared_ptr<Tensor> output;
        std::shared_ptr<Execution> runner;
    };
    ConvInt8Winograd(Backend *b, const Convolution2D *convOp, std::shared_ptr<ResourceInt8> res, std::vector<ConvInt8Winograd::UnitAttr>& unitAttrs);
    virtual ~ConvInt8Winograd();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static bool bestWinogradUnit(const Convolution2D *convOp, const Tensor *input, const Tensor* weightSrc, const Tensor *output, Backend* bn, std::vector<UnitAttr>& unitAttrs);
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    ConvInt8Winograd(Backend* backend, const Convolution2DCommon* common, const ConvInt8Winograd& exe);
    // transform func
    using WinoSrcTransFunc = WinogradInt8Helper::SrcTransFunc;
    using WinoDstTransFunc = WinogradInt8Helper::DstTransFunc;
    // subExecutions
    std::vector<Unit> mUnits;
    std::shared_ptr<CPUConvolution::ResourceInt8> mResource;
    
    class WinoExecution : public CPUConvolution {
    public:
        WinoExecution(Backend *b, const Convolution2DCommon* common, Tensor* weight, int unitY, int unitX, bool fastgemm);
        
        WinoExecution(Backend* bn, const Convolution2DCommon* common, const WinoExecution& exe);
        virtual ~WinoExecution();
        virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
        // weight
        std::shared_ptr<Tensor> mWeight;
        // buffer
        std::shared_ptr<Tensor> mTempInputBuffer;
        std::shared_ptr<Tensor> mTempOutputBuffer;
        std::shared_ptr<Tensor> mTransformMidBuffer;
        // transform func
        WinoSrcTransFunc mSourceTransformY = nullptr;
        WinoSrcTransFunc mSourceTransformX = nullptr;
        WinoDstTransFunc mDestTransformY = nullptr;
        WinoDstTransFunc mDestTransformX = nullptr;
        // unit and kernel
        int mUnitY, mUnitX;
        int mKernelY, mKernelX;
        // gemm func
        decltype(CoreInt8Functions::Int8GemmKernel) mGemmKernel;
        // other quan attr
        int8_t mInputZeroPoint;
        std::shared_ptr<Tensor> mOffsets;
        friend class ConvInt8Winograd;
    };
    
    static bool chooseTransformFuncs(int kernelY, int kernelX, int unitY, int unitX, ConvInt8Winograd::WinoExecution* exe, Backend* bn);
};
} // namespace MNN
#endif /* ConvInt8Winograd_hpp */
