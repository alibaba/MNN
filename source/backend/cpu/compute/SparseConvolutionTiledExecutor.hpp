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
namespace MNN {

// release resource when needed at different backend.
#define RELEASE_BUFFER_HINT                                              \
    if (nullptr != mNNZMap && nullptr != backend) {                      \
        backend->onReleaseBuffer(mNNZMap.get(), Backend::STATIC);        \
    }                                                                    \
    if (nullptr != mDataOffsetMap && nullptr != backend) {               \
        backend->onReleaseBuffer(mDataOffsetMap.get(), Backend::STATIC); \
    }

struct SparseIndexData {
    size_t sparseBlockOC;
    size_t weightNNZElement;
    size_t weightBlockNumber;
    Backend *backend;
    std::shared_ptr<Tensor> mNNZMap;
    std::shared_ptr<Tensor> mDataOffsetMap;
    SparseIndexData()
        : sparseBlockOC{0}, weightNNZElement{0}, weightBlockNumber{0}, backend{nullptr} {
    }
    SparseIndexData(size_t _sparseBlockOC, size_t _weightNNZElement, size_t _weightBlockNumber, Backend* _backend)
        : sparseBlockOC{_sparseBlockOC}, weightNNZElement{_weightNNZElement}, weightBlockNumber{_weightBlockNumber}, backend{_backend} {
    }
    SparseIndexData(const SparseIndexData& _sparseIndex) {
        sparseBlockOC = _sparseIndex.sparseBlockOC;
        weightNNZElement = _sparseIndex.weightNNZElement;
        weightBlockNumber = _sparseIndex.weightBlockNumber;
        backend = _sparseIndex.backend;
        mNNZMap = _sparseIndex.mNNZMap;
        mDataOffsetMap = _sparseIndex.mDataOffsetMap;
    }
    ~SparseIndexData() {
        // caution: in different backend, check resource is released or not.
    }
};

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
    SparseConvolutionTiledExecutor(const Convolution2DCommon *common, Backend *b, const IDSTQuan* weight, const SparseCommon* sparseCommon, const float *bias, size_t biasSize);

    SparseConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res, std::shared_ptr<SparseIndexData> mSparseIndexData,
                                  const Convolution2DCommon *common, MNNPackedSparseMatMul packedSparseMatmul, int sparseBlockOC, Backend *b);
    virtual ~SparseConvolutionTiledExecutor();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return mProxy->onExecute(inputs, outputs);
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        mInputs = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
        return mProxy->onResize(mInputs, outputs, mSparseIndexData->mNNZMap.get(), mSparseIndexData->mDataOffsetMap.get());
    }
    virtual bool onClone(Backend *bn, const Op *op, Execution **dst) override;

    void initWeight(float *dest, unsigned int *NNZMap, int *dataOffsetMap, int sparseBlockOC, const float *source,
                    const uint32_t* indexes, uint32_t indexSize, int depth, int outputCount, int kernelSize, int eP, size_t weightNNZElement,
                    size_t weightBlockNumber, const CoreFunctions *function);

protected:
    std::shared_ptr<SparseConvolutionTiledImpl> mProxy;
    std::shared_ptr<SparseIndexData> mSparseIndexData;
};

#undef RELEASE_BUFFER_HINT
} // namespace MNN

#endif /* SparseConvolutionTiledExecutor_hpp */
