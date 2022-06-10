//
//  ShapeTensorArray.cpp
//  MNN
//
//  Created by MNN on 2020/12/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <numeric>
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
    int totalSize = 0, arraySize = des->tensorArrayAttr->arraySize;
    for (auto elem : des->tensorArrayAttr->elemShape) {
        int elemSize = 1;
        for (auto dim : elem) {
            elemSize *= dim;
        }
        totalSize += elemSize;
    }
    if (des->tensorArrayAttr->elemShape.size() == 1 && arraySize > 1) {
        totalSize *= arraySize;
    } else if (totalSize == 0) {
        totalSize = 1; // bypass MNNV3 Dynamic Graph Executor zeroShape check
    }
    t->setLength(0, totalSize);
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
                    if (elemShape[i] < 0) {
                        elemShape[i] = 0;
                    }
                }
                des->tensorArrayAttr->elemShape.emplace_back(std::move(elemShape));
            }
            des->tensorArrayAttr->arraySize = inputs[0]->host<uint32_t>()[0];
            // 2. set dtype, dimension format and dims
            output->setType(param->T());
            TensorUtils::getDescribe(output)->dimensionFormat = op->defaultDimentionFormat();
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
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[1])->dimensionFormat;
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
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[2])->dimensionFormat;
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
        if (TensorUtils::getDescribe(inputs[2])->dimensionFormat != inDes->dimensionFormat) {
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
        if (TensorUtils::getDescribe(inputs[2])->dimensionFormat != inDes->dimensionFormat) {
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
        auto taParam = op->main_as_TensorArray();
        int splitAxis = (taParam->axis() + inputs[1]->dimensions()) % inputs[1]->dimensions();
        int keepdims = taParam->keepdims();
        copyTensorArrayAttribute(inputs[3], outputs[0]);
        outputs[0]->setType(op->main_as_TensorArray()->T());
        auto outDes = TensorUtils::getDescribe(outputs[0]);
        if (outDes->tensorArrayAttr->isIdenticalShape) {
            std::vector<int> writeElemShape(inputs[1]->shape());
            outDes->tensorArrayAttr->arraySize = writeElemShape[splitAxis];
            if (keepdims) {
                writeElemShape[splitAxis] = 1;
            } else {
                writeElemShape.erase(writeElemShape.begin() + splitAxis);
            }
            outDes->tensorArrayAttr->elemShape.emplace_back(std::move(writeElemShape));
        } else {
            auto value = inputs[1];
            auto lengths = inputs[2];
            bool scalarSplit = (lengths->elementSize() == 1);
            std::vector<int> vShape(value->shape());
            int totalLen = value->shape()[splitAxis], splitNum;
            if (scalarSplit) {
                splitNum = UP_DIV(totalLen, lengths->host<int>()[0]);
                MNN_ASSERT(keepdims || lengths->host<int>()[0] == 1);
            } else {
                splitNum = lengths->length(0);
                MNN_ASSERT(std::accumulate(lengths->host<int>(), lengths->host<int>() + splitNum, 0) == totalLen);
            }
            outDes->tensorArrayAttr->arraySize = splitNum;
            for (int i = 0; i < splitNum; ++i) {
                auto elemShape = vShape;
                if (scalarSplit) {
                    if (!keepdims) {
                        elemShape.erase(elemShape.begin() + splitAxis);
                    } else {
                        int splitLen = lengths->host<int>()[0];
                        elemShape[splitAxis] = ALIMIN(splitLen, totalLen - i * splitLen);
                    }
                } else {
                    elemShape[splitAxis] = lengths->host<int>()[i];
                }
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
        if (inDes->tensorArrayAttr == nullptr || inDes->tensorArrayAttr->arraySize == 0) {
            MNN_ASSERT(false);
            return false;
        }
        copyTensorArrayAttribute(inputs[1], outputs[0]);
        auto tpParam = op->main_as_TensorArray();
        int concatAxis = tpParam->axis(), newAxis = tpParam->new_axis();
        outputs[0]->setType(op->main_as_TensorArray()->T());

        const auto& elemShapes = inDes->tensorArrayAttr->elemShape;
        auto outShape = elemShapes[0];
        bool valid = true; // avoid use MNN_ASSERT because it's no-op in release mode
        for (int i = 1; valid && (i < elemShapes.size()); ++i) {
            auto elemShape = elemShapes[inDes->tensorArrayAttr->isIdenticalShape ? 0 : i];
            valid &= (outShape.size() == elemShape.size());
            if (newAxis) {
                valid &= (std::equal(outShape.begin(), outShape.end(), elemShape.begin()));
            } else {
                valid &= (std::equal(outShape.begin(), outShape.begin() + concatAxis, elemShape.begin()));
                valid &= (std::equal(outShape.begin() + concatAxis + 1, outShape.end(), elemShape.begin() + concatAxis + 1));
                outShape[concatAxis] += elemShape[concatAxis];
            }
        }
        if (!valid) {
            MNN_ERROR("Invalid input, elements in seq have different shape [new_axis=true need same shape, new_axis=false need same shape except concat_axis dim]\n");
            return false;
        }
        if (newAxis) {
            outShape.insert(outShape.begin() + concatAxis, inDes->tensorArrayAttr->arraySize);
        }
        outputs[0]->buffer().dimensions = outShape.size();
        for (int i = 0; i < outShape.size(); ++i) {
            outputs[0]->setLength(i, outShape[i]);
        }
        return true;
    }
};
REGISTER_SHAPE(TensorArrayConcatComputer, OpType_TensorArrayConcat);

// ============================ TensorArrayInsert ============================
class TensorArrayInsertComputer : public SizeComputer {
    // inputs : handle, position, value, flow_in
    // outputs: flow_out
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(4 == inputs.size() && 1 == outputs.size());
        auto inDes  = TensorUtils::getDescribe(inputs[3]);
        if (inDes->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        if (TensorUtils::getDescribe(inputs[2])->dimensionFormat != inDes->dimensionFormat) {
            MNN_ASSERT(false);
            return false;
        }
        MNN_ASSERT(inDes->tensorArrayAttr->isDynamicSize);

        copyTensorArrayAttribute(inputs[3], outputs[0]);
        auto outSeq = TensorUtils::getDescribe(outputs[0])->tensorArrayAttr;
        outputs[0]->buffer().type = inputs[3]->buffer().type;
        int inSeqSize = inDes->tensorArrayAttr->arraySize, insertIndex = inputs[1]->host<int32_t>()[0];
        MNN_ASSERT(insertIndex >= -inSeqSize && insertIndex <= inSeqSize); // [-n, n]
        insertIndex += (insertIndex < 0 ? inSeqSize : 0);
        // update arraySize
        outSeq->arraySize += 1;
        // update elemShape
        auto insertShape = inputs[2]->shape();
        auto& outSeqShapes = outSeq->elemShape;
        if (outSeq->isIdenticalShape && !outSeqShapes.empty()) {
            MNN_ASSERT(std::equal(insertShape.begin(), insertShape.end(), outSeqShapes[0].begin()));
        } else {
            outSeqShapes.insert(outSeqShapes.begin() + insertIndex, insertShape);
        }
        updateTensorArrayDims(outputs[0]);
        return true;
    }
};
REGISTER_SHAPE_INPUTS(TensorArrayInsertComputer, OpType_TensorArrayInsert, {1});

// ============================ TensorArrayErase ============================
class TensorArrayEraseComputer : public SizeComputer {
    // inputs : handle, position, flow_in
    // outputs: flow_out
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(3 == inputs.size() && 1 == outputs.size());
        auto inDes  = TensorUtils::getDescribe(inputs[2]);
        if (inDes->tensorArrayAttr == nullptr) {
            MNN_ASSERT(false);
            return false;
        }
        MNN_ASSERT(inDes->tensorArrayAttr->isDynamicSize);

        copyTensorArrayAttribute(inputs[2], outputs[0]);
        auto outSeq = TensorUtils::getDescribe(outputs[0])->tensorArrayAttr;
        outputs[0]->buffer().type = inputs[2]->buffer().type;
        int inSeqSize = outSeq->arraySize, eraseIndex = inputs[1]->host<int32_t>()[0];
        MNN_ASSERT(eraseIndex >= -inSeqSize && eraseIndex < inSeqSize); // [-n, n-1]
        eraseIndex += (eraseIndex < 0 ? inSeqSize : 0);
        // update arraySize
        outSeq->arraySize -= 1;
        // update elemShape
        if (!outSeq->isIdenticalShape) {
            outSeq->elemShape.erase(outSeq->elemShape.begin() + eraseIndex);
        }
        updateTensorArrayDims(outputs[0]);
        return true;
    }
};
REGISTER_SHAPE_INPUTS(TensorArrayEraseComputer, OpType_TensorArrayErase, {1});
} // namespace MNN
