//
//  BinaryOpOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <string>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(BinaryOpOnnx);

MNN::OpType BinaryOpOnnx::opType() {
    return MNN::OpType_BinaryOp;
}

MNN::OpParameter BinaryOpOnnx::type() {
    return MNN::OpParameter_BinaryOp;
}

void BinaryOpOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, OnnxScope* scope) {
    const auto &originalType = onnxNode->op_type();
    int inputSize = onnxNode->input_size();
    if (inputSize == 1) {
        DLOG(FATAL) << "Not support 1 input for " << originalType << " op";
        return;
    }
    std::vector<std::string> moreOps = {"Max", "Min", "Sum", "Mean"};
    if (inputSize > 2 && std::find(moreOps.begin(), moreOps.end(), originalType) == moreOps.end()) {
        DLOG(FATAL) << "Not support more than 2 input for " << originalType << " op";
        return;
    }

    void* param;
    if (inputSize == 2) {
        param = new MNN::BinaryOpT;
    } else {
        param = new MNN::ReductionParamT;
    }
#define TO_BINARY_OP(src, dst)       \
    if (originalType == src) {      \
        ((MNN::BinaryOpT*)param)->opType = dst; \
    }
#define TO_REDUCE_OP(src, dst) \
    if (originalType == src) {      \
        ((MNN::ReductionParamT*)param)->operation = dst; \
    }
#define TO_BINARY_OR_REDUCE_OP(src, dst0, dst1) \
    if (originalType == src) { \
        if (inputSize == 2) { \
            ((MNN::BinaryOpT*)param)->opType = dst0; \
        } else { \
            ((MNN::ReductionParamT*)param)->operation = dst1; \
        } \
    }
    
    TO_BINARY_OP("Add", MNN::BinaryOpOperation_ADD);
    TO_BINARY_OP("And", MNN::BinaryOpOperation_MUL);
    TO_BINARY_OP("Div", MNN::BinaryOpOperation_REALDIV);
    TO_BINARY_OP("Mul", MNN::BinaryOpOperation_MUL);
    TO_BINARY_OP("Equal", MNN::BinaryOpOperation_EQUAL);
    TO_BINARY_OP("Less", MNN::BinaryOpOperation_LESS);
    TO_BINARY_OP("LessOrEqual", MNN::BinaryOpOperation_LESS_EQUAL);
    TO_BINARY_OP("Greater", MNN::BinaryOpOperation_GREATER);
    TO_BINARY_OP("GreaterOrEqual", MNN::BinaryOpOperation_GREATER_EQUAL);
    TO_BINARY_OR_REDUCE_OP("Max", MNN::BinaryOpOperation_MAXIMUM, MNN::ReductionType_MAXIMUM);
    TO_BINARY_OR_REDUCE_OP("Min", MNN::BinaryOpOperation_MINIMUM, MNN::ReductionType_MINIMUM);
    if (originalType == "Mod") {
        int fmod = 0;
        for (const auto& attrProto : onnxNode->attribute()) {
            if (attrProto.name() == "fmod") {
                fmod = attrProto.i();
            }
        }
        ((MNN::BinaryOpT*)param)->opType = (fmod == 0 ? MNN::BinaryOpOperation_MOD : MNN::BinaryOpOperation_FLOORMOD);
    }
    TO_BINARY_OP("Pow", MNN::BinaryOpOperation_POW);
    TO_BINARY_OP("Sub", MNN::BinaryOpOperation_SUB);
    TO_BINARY_OR_REDUCE_OP("Sum", MNN::BinaryOpOperation_ADD, MNN::ReductionType_SUM);
    TO_REDUCE_OP("Mean", MNN::ReductionType_MEAN);
    TO_BINARY_OP("Or", MNN::BinaryOpOperation_LOGICALOR);
    TO_BINARY_OP("Xor", MNN::BinaryOpOperation_LOGICALXOR);
    
    if (originalType == "BitShift") {
        for (int i = 0; i < onnxNode->attribute_size(); ++i) {
            const auto& attributeProto = onnxNode->attribute(i);
            const auto& attributeName  = attributeProto.name();
            if (attributeName == "direction") {
                if (attributeProto.s() == "LEFT") {
                    ((MNN::BinaryOpT*)param)->opType = MNN::BinaryOpOperation_LEFTSHIFT;
                } else {
                    ((MNN::BinaryOpT*)param)->opType = MNN::BinaryOpOperation_RIGHTSHIFT;
                }
            }
        }
    }
    
    if (inputSize == 2) {
        dstOp->main.value = param;
        return;
    }
    
    // N input (i0, i1, ...) => 1 input (N, i0, i1, ...)
    std::unique_ptr<MNN::OpT> pack(new MNN::OpT);
    auto packName = dstOp->name + "/packed_input";
    pack->name = packName;
    pack->type = MNN::OpType_Pack;
    pack->main.type = MNN::OpParameter_PackParam;
    pack->main.value = new MNN::PackParamT;
    pack->main.AsPackParam()->axis = 0;
    pack->inputIndexes = dstOp->inputIndexes;
    int packedInput = scope->declareTensor(packName);
    pack->outputIndexes.assign({packedInput});
    scope->oplists().emplace_back(std::move(pack));
    
    // Reduce(Max/Min/Sum/Mean) along axis 0
    dstOp->type = MNN::OpType_Reduction;
    dstOp->main.type = MNN::OpParameter_ReductionParam;
    ((MNN::ReductionParamT*)param)->dim.assign({0});
    dstOp->main.value = param;
    dstOp->main.AsReductionParam()->keepDims = false;
    dstOp->main.AsReductionParam()->dim.assign({0});
    dstOp->inputIndexes.assign({packedInput});
}

REGISTER_CONVERTER(BinaryOpOnnx, Add);
REGISTER_CONVERTER(BinaryOpOnnx, And);
REGISTER_CONVERTER(BinaryOpOnnx, Sum);
REGISTER_CONVERTER(BinaryOpOnnx, Sub);
REGISTER_CONVERTER(BinaryOpOnnx, Div);
REGISTER_CONVERTER(BinaryOpOnnx, Mul);
REGISTER_CONVERTER(BinaryOpOnnx, Pow);
REGISTER_CONVERTER(BinaryOpOnnx, Equal);
REGISTER_CONVERTER(BinaryOpOnnx, Less);
REGISTER_CONVERTER(BinaryOpOnnx, LessOrEqual);
REGISTER_CONVERTER(BinaryOpOnnx, Greater);
REGISTER_CONVERTER(BinaryOpOnnx, GreaterOrEqual);
REGISTER_CONVERTER(BinaryOpOnnx, Max);
REGISTER_CONVERTER(BinaryOpOnnx, Min);
REGISTER_CONVERTER(BinaryOpOnnx, Mod);
REGISTER_CONVERTER(BinaryOpOnnx, Or);
REGISTER_CONVERTER(BinaryOpOnnx, Xor);
REGISTER_CONVERTER(BinaryOpOnnx, BitShift);
REGISTER_CONVERTER(BinaryOpOnnx, Mean);
