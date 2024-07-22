//
//  SparseConvInt8TiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2021/6/09.
//  Copyright © 2018 - 2021, Alibaba Group Holding Limited
//


#ifndef SparseConvInt8TiledExecutor_hpp
#define SparseConvInt8TiledExecutor_hpp
#include "ConvInt8TiledExecutor.hpp"
#include "backend/cpu/CPUConvolution.hpp"
#include "Int8FunctionsOpt.h"

#define SPARSITY_THRESHOLD (0.2f)

namespace MNN {


struct SparseQuantMatMulParam {
                    // only use size_t type
    size_t eSize;   // left matrix length of real value
    size_t eP;      // left matrix pack Unit
    size_t aStride; // left matrix stride
    size_t l;       // left matrix row, (kh * kw * ic/4 * 4)
    size_t h;       // right matrix colum, (oc)
    size_t cStride; // output matrix Stride on highest dim (ow * oh * C4Unit * bytes)
};

class SparseConvInt8TiledExecutor : public ConvInt8TiledExecutor {
public:
    // given weight+bias+scale, do post process
    SparseConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, std::shared_ptr<ResourceInt8> res);
    virtual ~SparseConvInt8TiledExecutor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    void getPackParameter(int* Unit, int* SrcUnit, int* DestUnit, const CoreInt8Functions* core) override;
    bool reorderWeight(Backend* b, const Convolution2DCommon* common, const std::shared_ptr<Tensor>& weightOrigin,
                       std::shared_ptr<Tensor>& weight, const SparseCommon* sparseCommon);

    static bool shouldUseSparse(const Convolution2D* conv2d) {
        auto common = conv2d->common();
        size_t originWeightSize = common->outputCount() * common->inputCount() * common->kernelY() * common->kernelX();
        const SparseCommon* sparseCommon = conv2d->sparseParameter();
        // MNN_PRINT("SparseConvInt8TiledExecutor sparsity:%f\n", 1 - float(sparseCommon->args()->LookupByKey("NNZElement")->i())/originWeightSize);
        return originWeightSize - sparseCommon->args()->LookupByKey("NNZElement")->i() >= originWeightSize * SPARSITY_THRESHOLD;
    }

private:
    SparseConvInt8TiledExecutor(Backend* backend, const Convolution2D* convOp, const SparseConvInt8TiledExecutor& exe);

    SparseQuantMatMulParam mSparseQuantParam;
    decltype(CoreInt8Functions::MNNPackedSparseQuantMatMulEpx1) mSparseQuantMatMulKernel;
    std::shared_ptr<Tensor> mNNZMap;
    std::shared_ptr<Tensor> mDataOffsetMap;
    int mSparseBlockOC;
};

} // namespace MNN

#undef SPARSITY_THRESHOLD

#endif /* SparseConvInt8TiledExecutor_hpp */
