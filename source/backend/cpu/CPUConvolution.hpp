//
//  CPUConvolution.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUConvolution_hpp
#define CPUConvolution_hpp

#include <mutex>
#include "CPUBackend.hpp"
#include "core/ConvolutionCommon.hpp"
namespace MNN {
    class PerfConfig {
public:
    PerfConfig() : isParallelInner{false}, eTile{1}, ePack{1}, hPack{1}, instructionCosts{.0f} {
    }
    PerfConfig(bool isParallelInner_, int eTile_, int ePack_, int hPack_, float instructionCosts_)
        : isParallelInner{isParallelInner_},  eTile{eTile_}, ePack{ePack_}, hPack{hPack_}, instructionCosts{instructionCosts_} {
    }
    bool operator!=(const PerfConfig& other) {
        return isParallelInner != other.isParallelInner || ePack != other.ePack || eTile != other.eTile || hPack != other.hPack;
    }
    PerfConfig& operator=(const PerfConfig& other) {
        isParallelInner = other.isParallelInner;
        ePack = other.ePack;
        eTile = other.eTile;
        hPack = other.hPack;
        instructionCosts = other.instructionCosts;
        return *this;
    }

    bool isParallelInner; // inner or outer parallel
    int eTile; // L2 cache tiling
    int ePack; // micro tile size along ow*oh dimension
    int hPack;
    float instructionCosts;
};
class CPUConvolution : public Execution {
public:
    struct Resource {
        std::shared_ptr<Tensor> mWeight;
        std::shared_ptr<Tensor> mBias;
        Backend* backend;
        bool copyBiasAlign(const float* bias, int outputCount);
        ~ Resource() {
            if (nullptr != mBias) {
                backend->onReleaseBuffer(mBias.get(), Backend::STATIC);
            }
            if (nullptr != mWeight) {
                backend->onReleaseBuffer(mWeight.get(), Backend::STATIC);
            }
        }
    };
    struct ResourceInt8 {
        std::vector<int> mInt8WeightKernelSum;
        std::shared_ptr<Tensor> mWeightInt8;
        std::shared_ptr<Tensor> mBiasInt32;
        std::shared_ptr<Tensor> mScaleFloat;
        // relu or relu6
        bool mRelu;
        int mActBits;

        int32_t mInputZeroPoint;
        int32_t mOutputZeroPoint;
        int8_t mClampMin;
        int8_t mClampMax;
        Backend* backend;
        float mInputScale;
        float mOutputScale;
        int mOutputCount;
#ifdef MNN_USE_SSE
        std::vector<int> offsets;
#endif
        std::once_flag flag;
        void updateInputOutputScale(std::vector<float> inputQuantInfo, std::vector<float> outputQuantInfo);
        ~ ResourceInt8();
    };
    static std::shared_ptr<ResourceInt8> makeResourceInt8(Backend *backend, const MNN::Convolution2D *convOp,
                                                          std::vector<float> inputQuantInfo, std::vector<float> outputQuantInfo);
    CPUConvolution(const Convolution2DCommon *convOp, Backend *b);
    virtual ~CPUConvolution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static int reorderWeightSize(int depth, int outputCount, int kernelSize, int unitDepth, int unitOC);
    // Inefficient but need not cache, use it when speed insensitive (init, onResize)
    // source shape: [outputCount, depth, kernelSize]
    // dest shape:
    // transpose=false: [UP_DIV(outputCount,unitOC), UP_DIV(depth,unitDepth), kernelSize, unitDepth, unitOC]
    // transpose=true:  [UP_DIV(outputCount,unitOC), UP_DIV(depth,unitDepth), kernelSize, unitOC, unitDepth]
    template<typename T> static void reorderWeightSlow(T* dest, const T* source, size_t depth, size_t outputCount, size_t kernelSize,
                                                       size_t unitDepth, size_t unitOC, bool transpose = false);
    /* Inefficient because of not use memcpy to support different type copy (T -> U), use it when speed insensitive (init, onResize)
       return: False if acquire failed
     */
    template<typename T, typename U> static bool acquireMemoryAndCopy(std::shared_ptr<Tensor> dest, const T* source, size_t count, Backend*);

    std::vector<float> getPostParameters() const;
public:
    PerfConfig mConvPerfconfig;
protected:
    const Convolution2DCommon *mCommon;
    // In execute, use pad from mPadX and mPadY, don't use mCommon's pad
    mutable int mPadX;
    mutable int mPadY;

};

} // namespace MNN

#endif /* CPUConvolution_hpp */
