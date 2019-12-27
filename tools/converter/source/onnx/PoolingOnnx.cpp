//
//  PoolingOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(PoolingOnnx);

MNN::OpType PoolingOnnx::opType() {
    return MNN::OpType_Pooling;
}
MNN::OpParameter PoolingOnnx::type() {
    return MNN::OpParameter_Pool;
}

static int poolSpatialDim(const onnx::NodeProto* onnxNode) {
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        if (attributeProto.name() != "kernel_shape") {
            continue;
        }
        return attributeProto.ints_size();
    }
    return -1;
}

static void runPooling3D(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         std::vector<const onnx::TensorProto*> initializers) {
    std::unique_ptr<MNN::Pool3DT> pool(new MNN::Pool3DT);
    const auto& type = onnxNode->op_type();
    if (type == "MaxPool") {
        pool->type = MNN::PoolType_MAXPOOL;
    } else if (type == "AveragePool") {
        pool->type = MNN::PoolType_AVEPOOL;
    } else {
        DLOG(ERROR) << "TODO ==> " << type;
    }
    pool->padType = MNN::PoolPadType_CAFFE;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        auto vec = std::vector<int>({
            static_cast<int>(attributeProto.ints(0)),
            static_cast<int>(attributeProto.ints(1)),
            static_cast<int>(attributeProto.ints(2))
        });
        if (attributeName == "kernel_shape") {
            pool->kernels = vec;
        } else if (attributeName == "strides") {
            pool->strides = vec;
        } else if (attributeName == "pads") {
            pool->pads = vec;
        }
    }
    dstOp->type = MNN::OpType_Pooling3D;
    dstOp->main.type = MNN::OpParameter_Pool3D;
    dstOp->main.value = pool.release();
}

void PoolingOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                      std::vector<const onnx::TensorProto*> initializers) {
    if (poolSpatialDim(onnxNode) == 3) {
        runPooling3D(dstOp, onnxNode, initializers);
        return;
    }
    auto poolParam  = new MNN::PoolT;
    int kw          = 1;
    int kh          = 1;
    int stride_h    = 1;
    int stride_w    = 1;
    int pad_w       = 0;
    int pad_h       = 0;
    bool ceil_model = false;

    const auto& type = onnxNode->op_type();

    if (type == "GlobalAveragePool") {
        poolParam->type     = MNN::PoolType_AVEPOOL;
        poolParam->isGlobal = true;
    } else if (type == "MaxPool" || type == "AveragePool") {
        if (type == "MaxPool") {
            poolParam->type = MNN::PoolType_MAXPOOL;
        } else {
            poolParam->type = MNN::PoolType_AVEPOOL;
        }
        poolParam->isGlobal = false;

        for (int i = 0; i < onnxNode->attribute_size(); ++i) {
            const auto& attributeProto = onnxNode->attribute(i);
            const auto& attributeName  = attributeProto.name();
            if (attributeName == "pads") {
                DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
                DCHECK(attributeProto.ints_size() == 4 || attributeProto.ints_size() == 2) << "Node Attribute ERROR";
                pad_h = attributeProto.ints(0);
                int pad_h_end;
                if (attributeProto.ints_size() == 2) {
                    pad_h_end = attributeProto.ints(1);
                } else {
                    pad_h_end     = attributeProto.ints(2);
                    pad_w         = attributeProto.ints(1);
                    int pad_w_end = attributeProto.ints(3);
                    DCHECK(pad_w == pad_w_end) << "Asymmetrical pads in pooling is not supported";
                }
                DCHECK(pad_h == pad_h_end) << "Asymmetrical pads in pooling is not supported";
            } else if (attributeName == "kernel_shape") {
                DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
                kh = attributeProto.ints(0);
                if (attributeProto.ints_size() == 2) {
                    kw = attributeProto.ints(1);
                }
            } else if (attributeName == "strides") {
                DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
                DCHECK(attributeProto.ints_size() == 2 || attributeProto.ints_size() == 1) << "Node Attribute ERROR";
                stride_h = attributeProto.ints(0);
                if (attributeProto.ints_size() == 2) {
                    stride_w = attributeProto.ints(1);
                }
            } else if (attributeName == "ceil_mode") {
                DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT) << "Node Attribute ERROR";
                ceil_model = static_cast<bool>(attributeProto.i());
            }
        }

    } else {
        DLOG(ERROR) << "TODO ==> " << type;
    }

    poolParam->kernelX   = kw;
    poolParam->kernelY   = kh;
    poolParam->strideX   = stride_w;
    poolParam->strideY   = stride_h;
    poolParam->padX      = pad_w;
    poolParam->padY      = pad_h;
    poolParam->dataType  = MNN::DataType_DT_FLOAT;
    poolParam->padType   = MNN::PoolPadType_CAFFE;
    poolParam->ceilModel = ceil_model;
    dstOp->main.value    = poolParam;
}

REGISTER_CONVERTER(PoolingOnnx, MaxPool);
REGISTER_CONVERTER(PoolingOnnx, GlobalAveragePool);
REGISTER_CONVERTER(PoolingOnnx, AveragePool);
