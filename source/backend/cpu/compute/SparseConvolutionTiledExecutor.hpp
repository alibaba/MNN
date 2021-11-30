//
//  SparseConvolutionTiledExecutor
//  MNN
//
//  Created by MNN on 2021/04/06.
//  Copyright © 2018-2021 Alibaba Group Holding Limited.
//

#ifndef SparseConvolutionTiledExecutor_hpp
#define SparseConvolutionTiledExecutor_hpp

#include <functional>
#include "backend/cpu/CPUConvolution.hpp"
#include "ConvolutionTiledExecutor.hpp"
// Tiled Slide Window or Im2Col + GEMM
#define SPARSITY_THRESHOLD (0.3f)
namespace MNN {

typedef void(*MNNPackedSparseMatMul)(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, unsigned int* NNZMap, int* dataOffsetMap);

class SparseConvolutionTiledImpl : public ConvolutionTiledImpl {
public:
    SparseConvolutionTiledImpl(const Convolution2DCommon *common, MNNPackedSparseMatMul packedSparseMatmul, int sparseBlockOC, Backend *b) : mPackedSparseMatmul{packedSparseMatmul}, mSparseBlockOC{sparseBlockOC}, ConvolutionTiledImpl(common, b) {

    }
    virtual ~SparseConvolutionTiledImpl() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, Tensor* NNZMap, Tensor* dataOffsetMap);
    void getPackParameter(int* eP, int* lP, int* hP, const CoreFunctions* core) override;
public:
    MNNPackedSparseMatMul mPackedSparseMatmul;
    int mSparseBlockOC;
};

class SparseConvolutionTiledExecutor : public ConvolutionTiledExecutor {
public:
    SparseConvolutionTiledExecutor(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                   size_t originWeightSize, const SparseCommon* sparseCommon, const float *bias, size_t biasSize);

    SparseConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res, std::shared_ptr<Tensor> NNZMapSharePtr, std::shared_ptr<Tensor> dataOffsetMapSharePtr,
                                  const Convolution2DCommon *common, MNNPackedSparseMatMul packedSparseMatmul, int sparseBlockOC, Backend *b);
    virtual ~SparseConvolutionTiledExecutor();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return mProxy->onExecute(inputs, outputs);
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        mInputs = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
        return mProxy->onResize(mInputs, outputs, mNNZMap.get(), mDataOffsetMap.get());
    }
    virtual bool onClone(Backend *bn, const Op *op, Execution **dst) override;

    void initWeight(float *dest, unsigned int *NNZMap, int *dataOffsetMap, int sparseBlockOC, const float *source,
                    float *cache, int depth, int outputCount, int kernelSize, int eP, size_t weightNNZElement,
                    size_t weightBlockNumber, const CoreFunctions *function);

    static bool shouldUseSparseConvolution(size_t originWeightSize, const SparseCommon* sparseCommon) {
        return originWeightSize - sparseCommon->args()->LookupByKey("NNZElement")->i() >= originWeightSize * SPARSITY_THRESHOLD;
    }
    static float getSparsityThreshold() {
        return SPARSITY_THRESHOLD;
    }
protected:
    std::shared_ptr<SparseConvolutionTiledImpl> mProxy;
    std::shared_ptr<Tensor> mNNZMap;
    std::shared_ptr<Tensor> mDataOffsetMap;
};
} // namespace MNN

#undef SPARSITY_THRESHOLD

#endif /* SparseConvolutionTiledExecutor_hpp */