//
//  SplitToSequenceOnnx.cpp
//  MNN
//
//  Created by MNN on 2019/06/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

// ======================= SplitToSequence =========================
DECLARE_OP_CONVERTER(SplitToSequenceOnnx);

MNN::OpType SplitToSequenceOnnx::opType() {
    return MNN::OpType_TensorArraySplit;
}
MNN::OpParameter SplitToSequenceOnnx::type() {
    return MNN::OpParameter_TensorArray;
}

void SplitToSequenceOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                              OnnxScope* scope) {
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = MNN::DataType_DT_FLOAT;
    tensorArray->keepdims = 1;// Default is 1 by onnx ref
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();

        if (attributeName == "axis") {
            tensorArray->axis = attributeProto.i();
        }
        if (attributeName == "keepdims") {
            tensorArray->keepdims = attributeProto.i();
        }
    }
    dstOp->main.value = tensorArray;
    
    // split(optional) -> split(required), because MNN Size Computer need explicit split index
    if (dstOp->inputIndexes.size() == 1) {
        dstOp->inputIndexes.push_back(scope->buildIntConstOp({1}, dstOp->name + "/split_default"));
    }
    auto tensorArrayIdx = scope->buildTensorArrayOp({}, false, dstOp->name + "/tensorArray");
    int valueIdx = dstOp->inputIndexes[0];
    int splitIdx = dstOp->inputIndexes[1];
    dstOp->inputIndexes.resize(4);
    // handle, value, lengths, flow_in
    dstOp->inputIndexes[0] = tensorArrayIdx.first;
    dstOp->inputIndexes[1] = valueIdx;
    dstOp->inputIndexes[2] = splitIdx;
    dstOp->inputIndexes[3] = tensorArrayIdx.second;
}

REGISTER_CONVERTER(SplitToSequenceOnnx, SplitToSequence);

// ======================= SequenceAt =========================
DECLARE_OP_CONVERTER(SequenceAtOnnx);
MNN::OpType SequenceAtOnnx::opType() {
    return MNN::OpType_TensorArrayRead;
}
MNN::OpParameter SequenceAtOnnx::type() {
    return MNN::OpParameter_TensorArray;
}
void SequenceAtOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         OnnxScope* scope) {
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = MNN::DataType_DT_FLOAT;
    dstOp->main.value = tensorArray;
    // handle, index, flow_in and handle == flow_in
    dstOp->inputIndexes.push_back(dstOp->inputIndexes[0]);
}
REGISTER_CONVERTER(SequenceAtOnnx, SequenceAt);

// ======================= SequenceLength =========================
DECLARE_OP_CONVERTER(SequenceLengthOnnx);
MNN::OpType SequenceLengthOnnx::opType() {
    return MNN::OpType_TensorArraySize;
}
MNN::OpParameter SequenceLengthOnnx::type() {
    return MNN::OpParameter_TensorArray;
}
void SequenceLengthOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         OnnxScope* scope) {
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = MNN::DataType_DT_FLOAT;
    dstOp->main.value = tensorArray;
    // handle, flow_in and handle == flow_in
    dstOp->inputIndexes.push_back(dstOp->inputIndexes[0]);
}
REGISTER_CONVERTER(SequenceLengthOnnx, SequenceLength);

// ======================= SequenceInsert =========================
DECLARE_OP_CONVERTER(SequenceInsertOnnx);
MNN::OpType SequenceInsertOnnx::opType() {
    return MNN::OpType_TensorArrayInsert;
}
MNN::OpParameter SequenceInsertOnnx::type() {
    return MNN::OpParameter_TensorArray;
}
void SequenceInsertOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         OnnxScope* scope) {
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = MNN::DataType_DT_FLOAT;
    dstOp->main.value = tensorArray;
    
    auto& indexs = dstOp->inputIndexes;
    // position(optional) -> position(required), because MNN SizeComputer need explcit position index
    if (indexs.size() == 2) {
        std::unique_ptr<MNN::OpT> op(new MNN::OpT);
        auto name = dstOp->name + "/seq_length";
        op->name = name;
        op->type = MNN::OpType_TensorArraySize;
        op->inputIndexes = {dstOp->inputIndexes[0], dstOp->inputIndexes[0]};
        op->outputIndexes.push_back(scope->declareTensor(name));
        indexs.push_back(op->outputIndexes[0]);
        scope->oplists().emplace_back(std::move(op));
    }
    // handle, tensor, position => handle, position, tensor, flow_in, for reusing inference code of OpType_TensorArrayWrite
    std::swap(indexs[1], indexs[2]);
    indexs.push_back(indexs[0]);
}
REGISTER_CONVERTER(SequenceInsertOnnx, SequenceInsert);

// ======================= SequenceErase =========================
DECLARE_OP_CONVERTER(SequenceEraseOnnx);
MNN::OpType SequenceEraseOnnx::opType() {
    return MNN::OpType_TensorArrayErase;
}
MNN::OpParameter SequenceEraseOnnx::type() {
    return MNN::OpParameter_TensorArray;
}
void SequenceEraseOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         OnnxScope* scope) {
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = MNN::DataType_DT_FLOAT;
    dstOp->main.value = tensorArray;
    
    auto& indexs = dstOp->inputIndexes;
    // position(optional) -> position(required), because MNN SizeComputer need explcit position index
    if (indexs.size() == 1) {
        // onnx default erase last element of seq, so index = seq.size() - 1
        std::unique_ptr<MNN::OpT> opSize(new MNN::OpT), opSub(new MNN::OpT);
        auto nameSize = dstOp->name + "/seq_length", nameSub = dstOp->name + "/seq_index";
        auto idxSize = scope->declareTensor(nameSize), idxSub = scope->declareTensor(nameSub);
        opSize->name = nameSize;
        opSize->type = MNN::OpType_TensorArraySize;
        opSize->inputIndexes.assign(2, indexs[0]); // handle, flow_in
        opSize->outputIndexes.push_back(idxSize);
        opSub->name = nameSub;
        opSub->type = MNN::OpType_BinaryOp;
        opSub->main.type = MNN::OpParameter_BinaryOp;
        auto paramSub = new MNN::BinaryOpT;
        paramSub->opType = MNN::BinaryOpOperation_SUB;
        opSub->main.value = paramSub;
        opSub->inputIndexes.assign({idxSize, scope->buildIntConstOp({1}, dstOp->name + "/const")});
        opSub->outputIndexes.push_back(idxSub);
        scope->oplists().emplace_back(std::move(opSize));
        scope->oplists().emplace_back(std::move(opSub));
        indexs.push_back(idxSub);
    }
    indexs.push_back(indexs[0]);
}
REGISTER_CONVERTER(SequenceEraseOnnx, SequenceErase);

// ======================= ConcatFromSequence =========================
DECLARE_OP_CONVERTER(ConcatFromSequenceOnnx);
MNN::OpType ConcatFromSequenceOnnx::opType() {
    return MNN::OpType_TensorArrayConcat;
}
MNN::OpParameter ConcatFromSequenceOnnx::type() {
    return MNN::OpParameter_TensorArray;
}
void ConcatFromSequenceOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         OnnxScope* scope) {
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = MNN::DataType_DT_FLOAT;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();

        if (attributeName == "axis") {
            tensorArray->axis = attributeProto.i();
        }
        if (attributeName == "new_axis") {
            tensorArray->new_axis = attributeProto.i();
        }
    }
    dstOp->main.value = tensorArray;
    // handle, flow_in and handle == flow_in
    dstOp->inputIndexes.push_back(dstOp->inputIndexes[0]);
}
REGISTER_CONVERTER(ConcatFromSequenceOnnx, ConcatFromSequence);

// ======================= SequenceConstruct =========================
DECLARE_OP_CONVERTER(SequenceConstructOnnx);
MNN::OpType SequenceConstructOnnx::opType() {
    return MNN::OpType_TensorArrayWrite;
}
MNN::OpParameter SequenceConstructOnnx::type() {
    return MNN::OpParameter_TensorArray;
}
void SequenceConstructOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         OnnxScope* scope) {
    auto tensorArray = new MNN::TensorArrayT;
    tensorArray->T = MNN::DataType_DT_FLOAT;
    dstOp->main.value = tensorArray;
    
    auto tensorArrayIdx = scope->buildTensorArrayOp({}, false, dstOp->name + "/tensorArray");
    int inputNum = dstOp->inputIndexes.size();
    for (int i = 0; i < inputNum; ++i) {
        auto op = (i == inputNum - 1 ? dstOp : new MNN::OpT);
        int insert_idx = scope->buildIntConstOp({i}, dstOp->name + "/tmp_index_" + std::to_string(i));
        int value_idx = dstOp->inputIndexes[i];
        op->inputIndexes.assign({tensorArrayIdx.first, insert_idx, value_idx, tensorArrayIdx.second});
        if (i < inputNum - 1) {
            auto name = dstOp->name + "/tmp_seq_" + std::to_string(i);
            op->name = name;
            op->type = MNN::OpType_TensorArrayWrite;
            op->main.type = MNN::OpParameter_TensorArray;
            auto tensorArray = new MNN::TensorArrayT;
            tensorArray->T = MNN::DataType_DT_FLOAT;
            op->main.value = tensorArray;
            int output_idx = scope->declareTensor(name);
            op->outputIndexes.assign({output_idx});
            scope->oplists().emplace_back(op);
            tensorArrayIdx.first = tensorArrayIdx.second = output_idx;
        }
    }
}
REGISTER_CONVERTER(SequenceConstructOnnx, SequenceConstruct);

// ======================= SequenceEmpty =========================
DECLARE_OP_CONVERTER(SequenceEmptyOnnx);
MNN::OpType SequenceEmptyOnnx::opType() {
    return MNN::OpType_Identity;
}
MNN::OpParameter SequenceEmptyOnnx::type() {
    return MNN::OpParameter_NONE;
}
void SequenceEmptyOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         OnnxScope* scope) {
    auto tensorArrayIdx = scope->buildTensorArrayOp({}, false, dstOp->name + "/tensorArray", 0);
    dstOp->inputIndexes.resize(1);
    dstOp->inputIndexes[0] = tensorArrayIdx.second;
}
REGISTER_CONVERTER(SequenceEmptyOnnx, SequenceEmpty);

// ======================= ReverseSequence =========================
DECLARE_OP_CONVERTER(ReverseSequenceOnnx);
MNN::OpType ReverseSequenceOnnx::opType() {
    return MNN::OpType_ReverseSequence;
}
MNN::OpParameter ReverseSequenceOnnx::type() {
    return MNN::OpParameter_ReverseSequenceParam;
}
void ReverseSequenceOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                         OnnxScope* scope) {
    
    int batchDim = 1, seqDim = 0;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "batch_axis") {
            batchDim = attributeProto.i();
        }
        if (attributeName == "time_axis") {
            seqDim = attributeProto.i();
        }
    }
    auto param = new MNN::ReverseSequenceParamT;
    param->batchDim = batchDim;
    param->seqDim = seqDim;
    dstOp->main.value = param;
}
REGISTER_CONVERTER(ReverseSequenceOnnx, ReverseSequence);
