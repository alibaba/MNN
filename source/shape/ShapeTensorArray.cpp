//
//  ShapeTensorArray.cpp
//  MNN
//
//  Created by MNN on 2020/12/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "math.h"

namespace MNN {
static void copyTensorArrayAttribute(const Tensor* src, Tensor* dst) {
    auto srcDes = TensorUtils::getDescribe(src);
    auto dstDes = TensorUtils::getDescribe(dst);
    dstDes->dimensionFormat = srcDes->dimensionFormat;
    dstDes->tensorArrayAttr.reset(new TensorArrayAttr);
    dstDes->tensorArrayAttr->isDynamicSize = srcDes->tensorArrayAttr->isDynamicSize;
    dstDes->tensorArrayAttr->isIdenticalShape = srcDes->tensorArrayAttr->isIdenticalShape;
    dstDes->tensorArrayAttr->arraySize = srcDes->tensorArrayAttr->arraySize;
    dstDes->tensorArrayAttr->elemShape = srcDes->tensorArrayAttr->elemShape;
}

static void updateTensorArrayDims(Tensor* t) {
    auto des = TensorUtils::getDescribe(t);
    // shape : [Sum(elemShape)]
    t->buffer().dimensions = 1;
    int totalSize = 0;
    for (auto elem : des->tensorArrayAttr->elemShape) {
        int elemSize = 1;
        for (auto dim : elem) {
            elemSize *= dim;
        }
        totalSize += elemSize;
    }
    t->setLength(0, des->tensorArrayAttr->arraySize * totalSize);
    t->setLength(1, 1);
    t->setLength(2, 1);
    t->setLength(3, 1);
}

// ============================ TensorArray ============================
class TensorArrayComputer : public SizeComputer {
    // inputs : size
    // outputs: handle, flow_out
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size() && 2 == outputs.size());
        auto param = op->main_as_TensorArray();
        for (int i = 0; i < 2; i++) {
            auto& output = outputs[i];
            auto des = TensorUtils::getDescribe(output);
            // 1. set TensorArray attrs
            des->tensorArrayAttr.reset(new TensorArrayAttr);
            des->tensorArrayAttr->isDynamicSize = param->dynamic_size();
            des->tensorArrayAttr->isIdenticalShape = param->identical_element_shapes();
            if (param->element_shape() && param->element_shape()->size() > 0) {
                std::vector<int> elemShape(param->element_shape()->size());
                for (int i = 0; i < param->element_shape()->size(); i++) {
                    elemShape[i] = param->element_shape()->Get(i);
                }
                des->tensorArrayAttr->elemShape.emplace_back(std::move(elemShape));
            }
            des->tensorArrayAttr->arraySize = inputs[0]->host<uint32_t>()[0];
            // 2. set dtype, dimension format and dims
            output->setType(param->T());
            TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
            updateTensorArrayDims(output);
            MNN_ASSERT(des->tensorArrayAttr != nullptr);
        }
        return true;
    }
};
REGISTER_SHAPE_INPUTS(TensorArrayComputer, OpType_TensorArray, {0});

// ============================ TensorArraySize ============================
class TensorArraySizeComputer : public SizeComputer {
    // inputs : handle, flow_in
    // outputs: tensor
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size() && 1 == outputs.size());
        MNN_ASSERT(TensorUtils::getDescribe(inputs[1])->tensorArrayAttr != nullptr);
        outputs[0]->setType(DataType_DT_INT32);
        outputs[0]->buffer().dimensions    = 1;
        outputs[0]->setLength(0, 1);
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        return true;
    }
};
REGISTER_SHAPE(TensorArraySizeComputer, OpType_TensorArraySize);

// ============================ TensorArrayRead ============================
class TensorArrayReadComputer : public SizeComputer {
    // inputs : handle, index, flow_in
    // outputs: tensor
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(3 == inputs.size() && 1 == outputs.size());
        auto des = TensorUtils::getDescribe(inputs[2]);
        if (des->tensorArrayAttr == nullptr) {
            return false;
        }
        std::vector<int> readElemShape;
        int readIndex = inputs[1]->host<uint32_t>()[0];
        if (!des->tensorArrayAttr->isIdenticalShape && des->tensorArrayAttr->elemShape.size() > readIndex) {
            readElemShape = des->tensorArrayAttr->elemShape[readIndex];
        } else if (des->tensorArrayAttr->elemShape.size() >= 1) {
            readElemShape = des->tensorArrayAttr->elemShape[0];
        } else {
            MNN_ASSERT(false);
        }
        outputs[0]->setType(op->main_as_TensorArray()->T());
        outputs[0]->buffer().dimensions    = readElemShape.size();
        for (int i = 0; i < readElemShape.size(); i++) {
            outputs[0]->setLength(i, readElemShape[i]);
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        return true;
    }
};
REGISTER_SHAPE_INPUTS(TensorArrayReadComputer, OpType_TensorArrayRead, {1});

// ============================ TensorArrayWrite ============================
class TensorArrayWriteComputer : public SizeComputer {
    // inputs : handle, index, value, flow_in
    // outputs: flow_out
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(4 == inputs.size() && 1 == outputs.size());
        auto inDes  = TensorUtils::getDescribe(inputs[3]);
        auto outDes = TensorUtils::getDescribe(outputs[0]);
        if (inDes->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        copyTensorArrayAttribute(inputs[3], outputs[0]);
        outputs[0]->setType(op->main_as_TensorArray()->T());
        int writeIndex = inputs[1]->host<uint32_t>()[0];
        // update arraySize
        if (!inDes->tensorArrayAttr->isDynamicSize) {
            MNN_ASSERT(writeIndex < inDes->tensorArrayAttr->arraySize);
        } else if (writeIndex >= inDes->tensorArrayAttr->arraySize) {
            outDes->tensorArrayAttr->arraySize = writeIndex + 1;
        }
        // update elemShape
        auto writeShape = inputs[2]->shape();
        if (outDes->tensorArrayAttr->isIdenticalShape) {
            if (outDes->tensorArrayAttr->elemShape.empty()) {
                outDes->tensorArrayAttr->elemShape.push_back(writeShape);
            } else {
                outDes->tensorArrayAttr->elemShape[0] = writeShape;
            }
        } else {
            for (int i = outDes->tensorArrayAttr->elemShape.size(); i <= writeIndex; i++) {
                outDes->tensorArrayAttr->elemShape.push_back(writeShape);
            }
            outDes->tensorArrayAttr->elemShape[writeIndex] = writeShape;
        }
        updateTensorArrayDims(outputs[0]);
        MNN_ASSERT(outDes->tensorArrayAttr != nullptr);
        return true;
    }
};
REGISTER_SHAPE_INPUTS(TensorArrayWriteComputer, OpType_TensorArrayWrite, {1});

// ============================ TensorArrayGather ============================
class TensorArrayGatherComputer : public SizeComputer {
    // inputs : handle, indices, flow_in
    // outputs: tensor
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(3 == inputs.size() && 1 == outputs.size());
        auto inDes  = TensorUtils::getDescribe(inputs[2]);
        auto outDes = TensorUtils::getDescribe(outputs[0]);
        if (inDes->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        auto param = op->main_as_TensorArray();
        outputs[0]->setType(param->T());
        outDes->dimensionFormat = inDes->dimensionFormat;
        outputs[0]->buffer().dimensions = inputs[2]->buffer().dimensions;
        outputs[0]->setLength(0, inputs[1]->length(0));
        // using param shape
        if (param->element_shape() && param->element_shape()->size() > 0) {
            outputs[0]->buffer().dimensions = param->element_shape()->size() + 1;
            MNN_ASSERT(param->element_shape()->size() == inDes->tensorArrayAttr->elemShape[0].size());
            for (int i = 0; i < param->element_shape()->size(); i++) {
                int dimValue = param->element_shape()->Get(i);
                if (dimValue < 0) {
                    dimValue = inDes->tensorArrayAttr->elemShape[0][i];
                }
                outputs[0]->setLength(1 + i, dimValue);
            }
        } else {
            if (inDes->tensorArrayAttr->elemShape.size() == 1) {
                for (int i = 0; i < inDes->tensorArrayAttr->elemShape[0].size(); i++) {
                    outputs[0]->setLength(1 + i, inDes->tensorArrayAttr->elemShape[0][i]);
                }
            } else {
                MNN_ASSERT(false);
            }
        }
        return true;
    }
};
REGISTER_SHAPE_INPUTS(TensorArrayGatherComputer, OpType_TensorArrayGather, {1});

// ============================ TensorArrayScatter ============================
class TensorArrayScatterComputer : public SizeComputer {
    // inputs : handle, indices, value, flow_in
    // outputs: flow_out
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(4 == inputs.size() && 1 == outputs.size());
        auto inDes  = TensorUtils::getDescribe(inputs[3]);
        auto outDes = TensorUtils::getDescribe(outputs[0]);
        if (inDes->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        copyTensorArrayAttribute(inputs[3], outputs[0]);
        for (int i = 0; i < inputs[1]->length(0); i++) {
            int writeIndex = inputs[1]->host<uint32_t>()[i];
            if (!inDes->tensorArrayAttr->isDynamicSize) {
                MNN_ASSERT(writeIndex < inDes->tensorArrayAttr->arraySize);
            } else if (writeIndex >= inDes->tensorArrayAttr->arraySize) {
                outDes->tensorArrayAttr->arraySize = writeIndex + 1;
            }
            std::vector<int> writeElemShape(inputs[2]->shape());
            writeElemShape.erase(writeElemShape.begin());
            if (outDes->tensorArrayAttr->elemShape.empty()) {
                outDes->tensorArrayAttr->elemShape.emplace_back(std::move(writeElemShape));
            } else {
                outDes->tensorArrayAttr->elemShape[0] = writeElemShape;
            }
        }
        outputs[0]->setType(op->main_as_TensorArray()->T());
        updateTensorArrayDims(outputs[0]);
        MNN_ASSERT(outDes->tensorArrayAttr != nullptr);
        return true;
    }
};
REGISTER_SHAPE_INPUTS(TensorArrayScatterComputer, OpType_TensorArrayScatter, {1});

// ============================ TensorArraySplit ============================
class TensorArraySplitComputer : public SizeComputer {
    // inputs : handle, value, lengths, flow_in
    // outputs: flow_out
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(4 == inputs.size() && 1 == outputs.size());
        auto inDes = TensorUtils::getDescribe(inputs[3]);
        if (inDes->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        copyTensorArrayAttribute(inputs[3], outputs[0]);
        outputs[0]->setType(op->main_as_TensorArray()->T());
        auto outDes = TensorUtils::getDescribe(outputs[0]);
        if (outDes->tensorArrayAttr->isIdenticalShape) {
            std::vector<int> writeElemShape(inputs[1]->shape());
            outDes->tensorArrayAttr->arraySize = writeElemShape[0];
            writeElemShape.erase(writeElemShape.begin());
            outDes->tensorArrayAttr->elemShape.emplace_back(std::move(writeElemShape));
        } else {
            auto value = inputs[1];
            auto lengths = inputs[2];
            outDes->tensorArrayAttr->arraySize = lengths->length(0);
            std::vector<int> vShape(value->shape());
            const int* lengthPtr = lengths->host<int>();
            for (int i = 0; i < lengths->length(0); i++) {
                auto elemShape = vShape;
                elemShape[0] = lengthPtr[i];
                outDes->tensorArrayAttr->elemShape.emplace_back(std::move(elemShape));
            }
        }
        updateTensorArrayDims(outputs[0]);
        MNN_ASSERT(outDes->tensorArrayAttr != nullptr);
        return true;
    }
};
REGISTER_SHAPE_INPUTS(TensorArraySplitComputer, OpType_TensorArraySplit, {2});

// ============================ TensorArrayConcat ============================
class TensorArrayConcatComputer : public SizeComputer {
    // inputs : handle, flow_in
    // outputs: tensor
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size() && 1 == outputs.size());
        auto inDes  = TensorUtils::getDescribe(inputs[1]);
        if (inDes->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        outputs[0]->setType(op->main_as_TensorArray()->T());
        if (inDes->tensorArrayAttr->elemShape.size() >= 1) {
            outputs[0]->buffer().dimensions = inDes->tensorArrayAttr->elemShape[0].size() + 1;
            outputs[0]->setLength(0, inDes->tensorArrayAttr->arraySize);
            for (int i = 0; i < inDes->tensorArrayAttr->elemShape[0].size(); i++) {
                outputs[0]->setLength(1 + i, inDes->tensorArrayAttr->elemShape[0][i]);
            }
        } else {
            MNN_ASSERT(false);
        }
        return true;
    }
};
REGISTER_SHAPE(TensorArrayConcatComputer, OpType_TensorArrayConcat);
} // namespace MNN
