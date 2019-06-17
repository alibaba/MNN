//
//  ConvolutionConverter.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionConverter.hpp"
#include "Macro.h"
#include "Tensor.hpp"
using namespace MNN;
static bool gZero = false;

static BlobT* createBlobFromTensor(const Tensor* tensor) {
    MNN_ASSERT(tensor->getType().code == halide_type_float);
    MNN_ASSERT(tensor->getDimensionType() == Tensor::TENSORFLOW);
    auto blob      = new BlobT;
    blob->dataType = DataType_DT_FLOAT;
    for (int i = 0; i < tensor->dimensions(); ++i) {
        blob->dims.emplace_back(tensor->length(i));
    }
    blob->float32s.resize(tensor->elementSize());
    ::memcpy(blob->float32s.data(), tensor->host<float>(), tensor->size());
    return blob;
}

OpConverter::Result ConvolutionConverter::onConvert(const MNN::OpT* op, const MNN::NetT* net) {
    OpConverter::Result result;
    auto conv2D            = op->main.AsConvolution2D();
    auto conv2DCommon      = conv2D->common.get();
    result.newTensorOffset = net->tensorName.size();
    if (op->inputIndexes.size() == 3) {
        return result;
    }
    {
        std::unique_ptr<OpT> weight(new OpT);
        weight->type      = OpType_Const;
        weight->name      = op->name + "_Filter";
        weight->main.type = OpParameter_Blob;
        auto srcCount     = (int)conv2D->weight.size() * conv2DCommon->group / conv2DCommon->outputCount /
                        conv2DCommon->kernelX / conv2DCommon->kernelY;
        weight->main.value                = new BlobT;
        weight->main.AsBlob()->dims       = {conv2DCommon->outputCount, conv2DCommon->kernelY, conv2DCommon->kernelX,
                                       srcCount / conv2DCommon->group};
        weight->main.AsBlob()->dataType   = DataType_DT_FLOAT;
        weight->main.AsBlob()->dataFormat = MNN_DATA_FORMAT_NHWC;
        weight->main.AsBlob()->float32s   = std::move(op->main.AsConvolution2D()->weight);
        if (gZero) {
            for (int i = 0; i < weight->main.AsBlob()->float32s.size(); ++i) {
                weight->main.AsBlob()->float32s[i] = ((int)(rand() % 100) - 50) / 10000.0f;
            }
        }
        weight->outputIndexes = {0 + result.newTensorOffset};
        result.opLists.emplace_back(std::move(weight));
        result.tensorNames.emplace_back(op->name + "_Filter");
        conv2DCommon->inputCount = srcCount;
    }
    {
        std::unique_ptr<OpT> weight(new OpT);
        weight->name          = op->name + "_Bias";
        weight->type          = OpType_Const;
        weight->main.type     = OpParameter_Blob;
        weight->outputIndexes = {(int)(1 + result.newTensorOffset)};
        std::shared_ptr<Tensor> biasTensor(Tensor::create<float>({conv2DCommon->outputCount}));
        ::memcpy(biasTensor->host<float>(), op->main.AsConvolution2D()->bias.data(),
                 conv2D->bias.size() * sizeof(float));
        weight->main.value = createBlobFromTensor(biasTensor.get());

        if (gZero) {
            for (int i = 0; i < weight->main.AsBlob()->float32s.size(); ++i) {
                weight->main.AsBlob()->float32s[i] = ((int)(rand() % 100) - 50) / 10000.0f;
            }
        }

        result.opLists.emplace_back(std::move(weight));
        result.tensorNames.emplace_back(op->name + "_Bias");
    }
    // Origin Convolution
    std::unique_ptr<OpT> newConvOp(new OpT);
    {
        newConvOp->type          = op->type;
        newConvOp->name          = op->name;
        newConvOp->inputIndexes  = {op->inputIndexes[0], 0 + result.newTensorOffset, 1 + result.newTensorOffset};
        newConvOp->outputIndexes = {op->outputIndexes[0]};
        newConvOp->main.type     = OpParameter_Convolution2D;
        newConvOp->main.value    = new Convolution2DT;
        newConvOp->main.AsConvolution2D()->common.reset(new Convolution2DCommonT(*conv2DCommon));
    }
    auto newConvOpPtr                                = newConvOp.get();
    if (conv2DCommon->padX != 0 && conv2DCommon->padMode == PadMode_CAFFE) {
        newConvOp->main.AsConvolution2D()->common->padMode = PadMode_SAME;
    }
    newConvOp->main.AsConvolution2D()->common->relu6 = false;
    newConvOp->main.AsConvolution2D()->common->relu  = false;
    result.opLists.emplace_back(std::move(newConvOp));

    // Seperate relu
    std::unique_ptr<OpT> nextOp;
    auto relu              = conv2DCommon->relu;
    auto relu6             = conv2DCommon->relu6;
    auto originOutputIndex = op->outputIndexes[0];
    if (relu) {
        nextOp.reset(new OpT);
        nextOp->name         = op->name + "_Relu";
        nextOp->type         = OpType_ReLU;
        nextOp->main.type    = OpParameter_Relu;
        auto reluT           = new ReluT;
        reluT->slope         = 0.0f;
        nextOp->main.value   = reluT;
        nextOp->inputIndexes = {2 + result.newTensorOffset};
        result.tensorNames.emplace_back(nextOp->name);
        nextOp->outputIndexes          = {originOutputIndex};
        newConvOpPtr->outputIndexes[0] = nextOp->inputIndexes[0];
        result.opLists.emplace_back(std::move(nextOp));
    } else if (relu6) {
        nextOp.reset(new OpT);
        nextOp->name         = op->name + "_Relu6";
        nextOp->type         = OpType_ReLU6;
        nextOp->inputIndexes = {2 + result.newTensorOffset};
        result.tensorNames.emplace_back(nextOp->name);
        nextOp->outputIndexes          = {originOutputIndex};
        newConvOpPtr->outputIndexes[0] = nextOp->inputIndexes[0];
        result.opLists.emplace_back(std::move(nextOp));
    }
    return result;
}

OpConverter::ReductResult ConvolutionConverter::onReduct(int opIndex, MNN::OpT* op, MNN::NetT* net) {
    OpConverter::ReductResult result;
    if (op->inputIndexes.size() != 3) {
        return result;
    }

    auto conv2D       = op->main.AsConvolution2D();
    auto conv2DCommon = conv2D->common.get();

    auto relu  = conv2DCommon->relu;
    auto relu6 = conv2DCommon->relu6;

    // set output
    auto outputIndex = op->outputIndexes[0];
    if (relu || relu6) {
        auto& reluOp = net->oplists[opIndex + 1];
        outputIndex  = reluOp->outputIndexes[0];
    }

    op->outputIndexes = {outputIndex};

    // add weight
    auto& weightOp                     = net->oplists[opIndex - 2];
    op->main.AsConvolution2D()->weight = std::move(weightOp->main.AsBlob()->float32s);

    // add bias
    auto& biasOp                     = net->oplists[opIndex - 1];
    op->main.AsConvolution2D()->bias = std::move(biasOp->main.AsBlob()->float32s);

    // set input
    op->inputIndexes = {op->inputIndexes[0]};

    result.needDeleteOpIndexes.emplace_back(opIndex - 2);
    result.needDeleteOpIndexes.emplace_back(opIndex - 1);
    if (relu || relu6) {
        result.needDeleteOpIndexes.emplace_back(opIndex + 1);
    }

    return result;
}

static const auto gRegister = []() {
    static ConvolutionConverter _c;
    OpConverter::insert(OpType_Convolution, &_c);
    OpConverter::insert(OpType_ConvolutionDepthwise, &_c);
    return true;
}();
