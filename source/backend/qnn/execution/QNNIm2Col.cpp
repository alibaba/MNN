//
//  QNNIm2Col.cpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNIm2Col.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNIm2Col::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto conv2D = mOp->main_as_Convolution2D();
    auto common = conv2D->common();

    int kernelH = common->kernelY();
    int kernelW = common->kernelX();
    int strideH = common->strideY();
    int strideW = common->strideX();
    int dilationH = common->dilateY();
    int dilationW = common->dilateX();
    int padY = common->padY();
    int padX = common->padX();

    auto input = inputs[0];
    auto output = outputs[0];

    // QNN Im2Col: input [batch, H, W, C] -> output [batch, C*kH*kW, L]
    // MNN Im2Col: output is 2D [batch*L, C*kH*kW]
    // We need a 3D stage tensor for QNN, then reshape to MNN's 2D output.
    int batch = input->length(0);
    int inH = input->length(1);
    int inW = input->length(2);
    int inC = input->length(3);
    int outH = (inH + 2 * padY - dilationH * (kernelH - 1) - 1) / strideH + 1;
    int outW = (inW + 2 * padX - dilationW * (kernelW - 1) - 1) / strideW + 1;
    int L = outH * outW;
    int colSize = inC * kernelH * kernelW;

    Qnn_DataType_t dataType = mBackend->getNativeTensor(input)->v1.dataType;

    std::vector<uint32_t> stageShape = {(uint32_t)batch, (uint32_t)colSize, (uint32_t)L};
    auto stageWrapper = this->createStageTensor("im2col_3d", dataType, stageShape);

    std::vector<uint32_t> kernelData = {(uint32_t)kernelH, (uint32_t)kernelW};
    std::vector<uint32_t> strideData = {(uint32_t)strideH, (uint32_t)strideW};
    std::vector<uint32_t> padAmountData = {(uint32_t)padY, (uint32_t)padY, (uint32_t)padX, (uint32_t)padX};
    std::vector<uint32_t> dilationData = {(uint32_t)dilationH, (uint32_t)dilationW};

    this->createParamTensor("kernel_size", QNN_DATATYPE_UINT_32, {2}, (void *)kernelData.data());
    this->createParamTensor("stride", QNN_DATATYPE_UINT_32, {2}, (void *)strideData.data());
    this->createParamTensor("pad_amount", QNN_DATATYPE_UINT_32, {2, 2}, (void *)padAmountData.data());
    this->createParamTensor("dilation", QNN_DATATYPE_UINT_32, {2}, (void *)dilationData.data());

    {
        CLEAR_BEFORE_ADDING_NODE;
        std::string name = mNodeName + "_Im2Col";
        mNodeType = "Im2Col";
        for (int i = 0; i < mParamTensorWrappers.size(); i++) {
            mParams.push_back(*(mParamTensorWrappers[i]->getNativeParam()));
        }
        mInputs.push_back(*(mBackend->getNativeTensor(input)));
        mOutputs.push_back(*(stageWrapper->getNativeTensor()));
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    this->addNodeCommonReshape("Reshape", *(stageWrapper->getNativeTensor()), *(mBackend->getNativeTensor(output)));

    return NO_ERROR;
}


class QNNIm2ColCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                                Backend* backend) const override {
        return new QNNIm2Col(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNIm2ColCreator, OpType_Im2Col)
#endif
} // end namespace QNN
} // end namespace MNN
