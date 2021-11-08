//
//  TensorArrayTf.cpp
//  MNNConverter
//
//  Created by MNN on 2020/12/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TfUtils.hpp"
#include "tfOpConverter.hpp"

#include "graph.pb.h"

// ============================ TensorArray ============================
DECLARE_OP_CONVERTER(TensorArrayTf);

MNN::OpType TensorArrayTf::opType() {
    return MNN::OpType_TensorArray;
}
MNN::OpParameter TensorArrayTf::type() {
    return MNN::OpParameter_TensorArray;
}

void TensorArrayTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto tensorArray = new MNN::TensorArrayT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "dtype", value)) {
        tensorArray->T = (MNN::DataType)value.type();
    }
    if (find_attr_value(srcNode->tfNode, "dynamic_size", value)) {
        tensorArray->dynamic_size = value.b();
    }
    if (find_attr_value(srcNode->tfNode, "identical_element_shapes", value)) {
        tensorArray->identical_element_shapes = value.b();
    }
    if (find_attr_value(srcNode->tfNode, "element_shape", value)) {
        if (value.shape().dim_size() > 0) {
            tensorArray->element_shape.resize(value.shape().dim_size());
            for (int i = 0; i < value.shape().dim_size(); i++) {
                tensorArray->element_shape[i] = value.shape().dim(i).size();
            }
        }
    }
    dstOp->main.value = tensorArray;
}

REGISTER_CONVERTER(TensorArrayTf, TensorArrayV3);

// ============================ TensorArraySize ============================
DECLARE_OP_CONVERTER(TensorArraySizeTf);

MNN::OpType TensorArraySizeTf::opType() {
    return MNN::OpType_TensorArraySize;
}
MNN::OpParameter TensorArraySizeTf::type() {
    return MNN::OpParameter_NONE;
}

void TensorArraySizeTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    dstOp->main.value = nullptr;
}

REGISTER_CONVERTER(TensorArraySizeTf, TensorArraySizeV3);


// ============================ TensorArrayRead ============================
DECLARE_OP_CONVERTER(TensorArrayReadTf);

MNN::OpType TensorArrayReadTf::opType() {
    return MNN::OpType_TensorArrayRead;
}
MNN::OpParameter TensorArrayReadTf::type() {
    return MNN::OpParameter_TensorArray;
}

void TensorArrayReadTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto tensorArrayRead = new MNN::TensorArrayT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "dtype", value)) {
        tensorArrayRead->T = (MNN::DataType)value.type();
    }
    dstOp->main.value = tensorArrayRead;
}

REGISTER_CONVERTER(TensorArrayReadTf, TensorArrayReadV3);

// ============================ TensorArrayWrite ============================
DECLARE_OP_CONVERTER(TensorArrayWriteTf);

MNN::OpType TensorArrayWriteTf::opType() {
    return MNN::OpType_TensorArrayWrite;
}
MNN::OpParameter TensorArrayWriteTf::type() {
    return MNN::OpParameter_TensorArray;
}

void TensorArrayWriteTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto tensorArrayWrite = new MNN::TensorArrayT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "T", value)) {
        tensorArrayWrite->T = (MNN::DataType)value.type();
    }
    dstOp->main.value = tensorArrayWrite;
}

REGISTER_CONVERTER(TensorArrayWriteTf, TensorArrayWriteV3);

// ============================ TensorArrayGather ============================
DECLARE_OP_CONVERTER(TensorArrayGatherTf);

MNN::OpType TensorArrayGatherTf::opType() {
    return MNN::OpType_TensorArrayGather;
}
MNN::OpParameter TensorArrayGatherTf::type() {
    return MNN::OpParameter_TensorArray;
}

void TensorArrayGatherTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto tensorArrayGather = new MNN::TensorArrayT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "dtype", value)) {
        tensorArrayGather->T = (MNN::DataType)value.type();
    }
    if (find_attr_value(srcNode->tfNode, "element_shape", value)) {
        if (value.shape().dim_size() > 0) {
            tensorArrayGather->element_shape.resize(value.shape().dim_size());
            for (int i = 0; i < value.shape().dim_size(); i++) {
                tensorArrayGather->element_shape[i] = value.shape().dim(i).size();
            }
        }
    }
    dstOp->main.value = tensorArrayGather;
}

REGISTER_CONVERTER(TensorArrayGatherTf, TensorArrayGatherV3);

// ============================ TensorArrayScatter ============================
DECLARE_OP_CONVERTER(TensorArrayScatterTf);

MNN::OpType TensorArrayScatterTf::opType() {
    return MNN::OpType_TensorArrayScatter;
}
MNN::OpParameter TensorArrayScatterTf::type() {
    return MNN::OpParameter_TensorArray;
}

void TensorArrayScatterTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto tensorArrayScatter = new MNN::TensorArrayT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "T", value)) {
        tensorArrayScatter->T = (MNN::DataType)value.type();
    }
    dstOp->main.value = tensorArrayScatter;
}

REGISTER_CONVERTER(TensorArrayScatterTf, TensorArrayScatterV3);

// ============================ TensorArraySplit ============================
DECLARE_OP_CONVERTER(TensorArraySplitTf);

MNN::OpType TensorArraySplitTf::opType() {
    return MNN::OpType_TensorArraySplit;
}
MNN::OpParameter TensorArraySplitTf::type() {
    return MNN::OpParameter_TensorArray;
}

void TensorArraySplitTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto tensorArraySplit = new MNN::TensorArrayT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "T", value)) {
        tensorArraySplit->T = (MNN::DataType)value.type();
    }
    dstOp->main.value = tensorArraySplit;
}

REGISTER_CONVERTER(TensorArraySplitTf, TensorArraySplitV3);

// ============================ TensorArrayConcat ============================
DECLARE_OP_CONVERTER(TensorArrayConcatTf);

MNN::OpType TensorArrayConcatTf::opType() {
    return MNN::OpType_TensorArrayConcat;
}
MNN::OpParameter TensorArrayConcatTf::type() {
    return MNN::OpParameter_TensorArray;
}

void TensorArrayConcatTf::run(MNN::OpT *dstOp, TmpNode *srcNode) {
    auto tensorArrayConcat = new MNN::TensorArrayT;
    tensorflow::AttrValue value;
    if (find_attr_value(srcNode->tfNode, "T", value)) {
        tensorArrayConcat->T = (MNN::DataType)value.type();
    }
    if (find_attr_value(srcNode->tfNode, "element_shape", value)) {
        if (value.shape().dim_size() > 0) {
            tensorArrayConcat->element_shape.resize(value.shape().dim_size());
            for (int i = 0; i < value.shape().dim_size(); i++) {
                tensorArrayConcat->element_shape[i] = value.shape().dim(i).size();
            }
        }
    }
    dstOp->main.value = tensorArrayConcat;
}

REGISTER_CONVERTER(TensorArrayConcatTf, TensorArrayConcatV3);
