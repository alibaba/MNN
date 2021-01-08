//
//  CPUConvolution.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConvolution_hpp
#define CPUConvolution_hpp

#include "CPUBackend.hpp"
#include "core/ConvolutionCommon.hpp"
namespace MNN {
class CPUConvolution : public Execution {
public:
    struct Resource {
        std::shared_ptr<Tensor> mWeight;
        std::shared_ptr<Tensor> mBias;
        Backend* backend;
        ~ Resource() {
            if (nullptr != mBias) {
                backend->onReleaseBuffer(mBias.get(), Backend::STATIC);
            }
            if (nullptr != mWeight) {
                backend->onReleaseBuffer(mWeight.get(), Backend::STATIC);
            }
        }
    };
    CPUConvolution(const Convolution2DCommon *convOp, Backend *b);
    virtual ~CPUConvolution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    typedef void (*POSTFUNCTION)(float *dst, const float *bias, size_t planeNumber, size_t biasNumber);

    POSTFUNCTION getPostFunction() const;
    
    static int reorderWeightSize(int depth, int outputCount, int kernelSize, int unitDepth, int unitOC);
    // Inefficient but need not cache, use it when speed insensitive (init, onResize)
    template<typename T> static void reorderWeightSlow(T* dest, const T* source, size_t depth, size_t outputCount, size_t kernelSize,
                                                       size_t unitDepth, size_t unitOC, bool transpose = false);
    /* Inefficient because of not use memcpy to support different type copy (T -> U), use it when speed insensitive (init, onResize)
       return: False if acquire failed
     */
    template<typename T, typename U> static bool acquireMemoryAndCopy(std::shared_ptr<Tensor> dest, const T* source, size_t count, Backend*);

    std::vector<float> getPostParameters() const;
protected:
    const Convolution2DCommon *mCommon;

    // In execute, use pad from mPadX and mPadY, don't use mCommon's pad
    mutable int mPadX;
    mutable int mPadY;
    CPUConvolution::POSTFUNCTION mPostFunction;
};

} // namespace MNN

#endif /* CPUConvolution_hpp */
