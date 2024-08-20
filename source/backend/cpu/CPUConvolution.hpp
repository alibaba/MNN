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
    struct ResourceDequantizeInfo {
        int bits = 32;
        std::shared_ptr<Tensor> mScaleBias;
    };
    struct Resource {
        std::shared_ptr<Tensor> mWeightKernelSum;
        std::shared_ptr<Tensor> mWeight;
        std::shared_ptr<Tensor> mBias;
        ResourceDequantizeInfo mDequantize;
        Backend* backend;
        bool copyBiasAlign(const float* bias, int outputCount);
        int hU;
        int lU;
        int lP;
        int hP;
        std::vector<float> mReluThreshold;
    };
    struct ResourceInt8 {
        std::vector<int> mInt8WeightKernelSum;
        std::shared_ptr<Tensor> mWeightInt8;
        std::shared_ptr<Tensor> mOriginBias;
        std::shared_ptr<Tensor> mOriginScale;
        std::shared_ptr<Tensor> mWeightQuantZero;
        // relu or relu6
        bool mRelu;
        int mActBits;

        int mOutputCount;
        bool mUseConvQuan = true;
        bool mWeightAsymmetricQuant = true;
#ifdef MNN_USE_SSE
        std::vector<int> offsets;
#endif
        // Origin Attributes from net
        float mInputScale = 0.0f;
        float mOutputScale = 0.0f;
        int32_t mInputZeroPoint;
        int32_t mOutputZeroPoint;
        int8_t mClampMin;
        int8_t mClampMax;
    };
    struct MutableResourceInt8 {
        MutableResourceInt8(std::shared_ptr<ResourceInt8> res, Backend* backend);
        void updateInputOutputScale(std::vector<float> inputQuantInfo, std::vector<float> outputQuantInfo);
        std::shared_ptr<ResourceInt8> mResource;
        float mInputScale = 0.0f;
        float mOutputScale = 0.0f;
        int32_t mInputZeroPoint;
        int32_t mOutputZeroPoint;
        int8_t mClampMin;
        int8_t mClampMax;
        std::shared_ptr<Tensor> mBiasInt32;
        std::shared_ptr<Tensor> mScaleFloat;
        std::shared_ptr<Tensor> mBiasFloat;
        int32_t mShiftBits = 14;
        bool mValid;
    };
    static std::shared_ptr<ResourceInt8> makeResourceInt8(Backend *backend, const MNN::Convolution2D *convOp, int pack=4);
    static void makeResource(Backend* backend, std::shared_ptr<Resource> resource, const Convolution2D* conv2d, std::shared_ptr<ResourceInt8> resourceInt8 = nullptr);
    CPUConvolution(const Convolution2DCommon *convOp, Backend *b);
    virtual ~CPUConvolution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static int reorderWeightSize(int depth, int outputCount, int kernelSize, int unitDepth, int unitOC);

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
