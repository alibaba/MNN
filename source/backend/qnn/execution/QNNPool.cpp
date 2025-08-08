//
//  QNNPool.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNPool.hpp"

namespace MNN {
namespace QNN {

ErrorCode QNNPool::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Params: filter_size([h, w]), stride([h, w]), pad_amount([[height_pad_before, height_pad_after], [width_pad_before, width_pad_after]]), count_pad_for_edges(false), rounding_mode

    mParams.clear();
    mInputs.clear();
    mOutputs.clear();

    if (mOp->type() == OpType_Pooling3D) {
        return this->onEncode3D(inputs, outputs);
    }

    mNodeType = "PoolAvg2d";
    std::vector<uint32_t> filterSizeData(2);
    std::vector<uint32_t> strideData(2);
    std::vector<uint32_t> padAmountData(4);
    uint32_t roundingMode;

    setParamPool(mNodeType, filterSizeData, strideData, padAmountData, roundingMode, inputs[0], outputs[0]);

    // shape(out[0])[height_out] = ROUND((pad_amount[0,0] + shape(in[0])[height] + pad_amount[0,1] - filter_size[0]) / stride[0] + 1)
    if(inputs[0]->height() < filterSizeData[0]) {
        filterSizeData[0] = inputs[0]->height();
    }
    if(inputs[0]->width() < filterSizeData[1]) {
        filterSizeData[1] = inputs[0]->width();
    }
    this->createParamTensor("filter_size", QNN_DATATYPE_UINT_32, {2}, (void *)filterSizeData.data());
    this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2}, (void *)strideData.data());
    this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2}, (void *)padAmountData.data());

    if (mOp->main_as_Pool()->type() == PoolType_AVEPOOL) {
        bool countType = mOp->main_as_Pool()->countType() ? true : false;
        this->createParamScalar("count_pad_for_edges", countType);
    }
    this->createParamScalar("rounding_mode", roundingMode);

    #ifdef QNN_VERBOSE
    MNN_PRINT("QNN Pool input:");
    auto shape0 = inputs[0]->shape();
    for(int i = 0; i < shape0.size(); i++) {
        MNN_PRINT("%d x ", shape0[i]);
    }

    MNN_PRINT("\noutput:");
    auto outShape = outputs[0]->shape();
    for(int i = 0; i < outShape.size(); i++) {
        MNN_PRINT("%d x ", outShape[i]);
    }
    MNN_PRINT("\n");
    MNN_PRINT("mNodeType:%s, filterSizeData:%dx%d, strideData:%dx%d, padAmountData:%dx%dx%dx%d, roundingMode:%d\n", mNodeType.c_str(), \
        filterSizeData[0], filterSizeData[1], strideData[0], strideData[1], padAmountData[0], padAmountData[1], padAmountData[2], padAmountData[3], roundingMode);
    #endif

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}


ErrorCode QNNPool::onEncode3D(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    if (input->dimensions() != 4) {
        MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }
    auto pool3D = mOp->main_as_Pool3D();
    mNodeType = (pool3D->type() == PoolType_AVEPOOL) ? "PoolAvg2d" : "PoolMax2d";

    std::vector<uint32_t> filterSizeData(2);
    std::vector<uint32_t> strideData(2);
    std::vector<uint32_t> padAmountData(4);
    uint32_t roundingMode;

    uint32_t * inputDim = mBackend->getNativeTensor(inputs[0])->v1.dimensions;
    uint32_t height = inputDim[1];
    uint32_t width = inputDim[2];

    if (pool3D->isGlobal()) {
        filterSizeData[0] = height;
        filterSizeData[1] = width;
        strideData[0] = height;
        strideData[1] = width;
        padAmountData[0] = 0;
        padAmountData[1] = 0;
        padAmountData[2] = 0;
        padAmountData[3] = 0;
        roundingMode = 1; // <ceil> or <floor> makes no difference.
    } else {
        MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }

    this->createParamTensor("filter_size", QNN_DATATYPE_UINT_32, {2}, (void *)filterSizeData.data());
    this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2}, (void *)strideData.data());
    this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2}, (void *)padAmountData.data());

    if (pool3D->type() == PoolType_AVEPOOL) {
        // bool countType = mOp->main_as_Pool()->countType() ? true : false;
        this->createParamScalar("count_pad_for_edges", false);
    }
    this->createParamScalar("rounding_mode", roundingMode);

    this->addNodeCommon(inputs, outputs);

    return NO_ERROR;
}


void QNNPool::setParamPool(std::string & nodeType, std::vector<uint32_t> & filterSizeData, std::vector<uint32_t> & strideData, std::vector<uint32_t> & padAmountData, uint32_t & roundingMode, Tensor * input, Tensor * output) {
    auto pool = mOp->main_as_Pool();
    nodeType = (pool->type() == PoolType_AVEPOOL) ? "PoolAvg2d" : "PoolMax2d";

    if (pool->isGlobal()) {
        filterSizeData[0] = input->height();
        filterSizeData[1] = input->width();
        strideData[0] = input->height();
        strideData[1] = input->width();
        padAmountData[0] = 0;
        padAmountData[1] = 0;
        padAmountData[2] = 0;
        padAmountData[3] = 0;
        roundingMode = 1; // <ceil> or <floor> makes no difference.
        return;
    }

    filterSizeData[0] = pool->kernelY();
    filterSizeData[1] = pool->kernelX();
    strideData[0] = pool->strideY();
    strideData[1] = pool->strideX();

    if (pool->padType() == PoolPadType_SAME) {
        int padNeededWidth  = (output->width() - 1) * strideData[1] + filterSizeData[1] - input->width();
        int padNeededHeight = (output->height() - 1) * strideData[0] + filterSizeData[0] - input->height();
        auto padLeft = padNeededWidth / 2;
        auto padTop = padNeededHeight / 2;

        auto padRight = padNeededWidth - padLeft;
        auto padBottom = padNeededHeight - padTop;

        padAmountData[0] = padTop;
        padAmountData[1] = padBottom;
        padAmountData[2] = padLeft;
        padAmountData[3] = padRight;
        roundingMode = 1; // ceil
        return;
    }

    if (pool->padType() == PoolPadType_VALID) {
        padAmountData[0] = 0;
        padAmountData[1] = 0;
        padAmountData[2] = 0;
        padAmountData[3] = 0;
        roundingMode = 0; // floor
        return;
    }

    if (nullptr != pool->pads()) {
        MNN_ASSERT(pool->pads()->size() == 4);
        padAmountData[0] = pool->pads()->data()[0];
        padAmountData[1] = pool->pads()->data()[2];
        padAmountData[2] = pool->pads()->data()[1];
        padAmountData[3] = pool->pads()->data()[3];
    } else {
        padAmountData[0] = pool->padY();
        padAmountData[1] = pool->padY();
        padAmountData[2] = pool->padX();
        padAmountData[3] = pool->padX();
    }
    roundingMode = (pool->ceilModel()) ? 1 : 0;

    return;
}


class QNNPoolCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNPool(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNPoolCreator, OpType_Pooling)
REGISTER_QNN_OP_CREATOR(QNNPoolCreator, OpType_Pooling3D)

} // end namespace QNN
} // end namespace MNN

