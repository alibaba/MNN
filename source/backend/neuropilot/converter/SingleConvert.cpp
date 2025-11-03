#include "SingleConvert.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
static tflite::BuiltinOperator getBinaryOpTFLiteType(int mnnBinaryOpType) {
    switch (mnnBinaryOpType) {
        case BinaryOpOperation_ADD:
            return tflite::BuiltinOperator_ADD;
        case BinaryOpOperation_SUB:
            return tflite::BuiltinOperator_SUB;
        case BinaryOpOperation_MUL:
            return tflite::BuiltinOperator_MUL;
        case BinaryOpOperation_DIV:
        case BinaryOpOperation_REALDIV:
            return tflite::BuiltinOperator_DIV;
        case BinaryOpOperation_MINIMUM:
            return tflite::BuiltinOperator_MINIMUM;
        case BinaryOpOperation_MAXIMUM:
            return tflite::BuiltinOperator_MAXIMUM;
        case BinaryOpOperation_EQUAL:
            return tflite::BuiltinOperator_EQUAL;
        case BinaryOpOperation_LESS:
            return tflite::BuiltinOperator_LESS;
        case BinaryOpOperation_LESS_EQUAL:
            return tflite::BuiltinOperator_LESS_EQUAL;
        case BinaryOpOperation_GREATER:
            return tflite::BuiltinOperator_GREATER;
        case BinaryOpOperation_GREATER_EQUAL:
            return tflite::BuiltinOperator_GREATER_EQUAL;
        case BinaryOpOperation_NOTEQUAL:
            return tflite::BuiltinOperator_NOT_EQUAL;
        case BinaryOpOperation_POW:
            return tflite::BuiltinOperator_POW;
        case BinaryOpOperation_FLOORDIV:
            return tflite::BuiltinOperator_FLOOR_DIV;
        default:
            MNN_PRINT("Warning: Unsupported MNN BinaryOpOperation %d, using ADD\n", mnnBinaryOpType);
            return tflite::BuiltinOperator_ADD;
    }
}
static tflite::BuiltinOperator _getEltwiseOp(const MNN::Op* op) {
    auto elt = op->main_as_Eltwise();
    switch (elt->type()) {
        case EltwiseType_SUM:
            return tflite::BuiltinOperator_ADD;
        case EltwiseType_SUB:
            return tflite::BuiltinOperator_SUB;
        case EltwiseType_PROD:
            return tflite::BuiltinOperator_MUL;
        default:
            break;
    }
    return tflite::BuiltinOperator_ADD;
}
static tflite::BuiltinOperator _mapMNNOpToTFLiteOp(const MNN::Op* op) {
    auto mnnOpType = op->type();
    switch (mnnOpType) {
        case OpType_FloatToInt8:
            return tflite::BuiltinOperator_QUANTIZE;
        case OpType_Int8ToFloat:
            return tflite::BuiltinOperator_DEQUANTIZE;
        case OpType_StridedSlice:
            return tflite::BuiltinOperator_STRIDED_SLICE;
        case OpType_Gather:
        case OpType_GatherV2:
            return tflite::BuiltinOperator_GATHER;
        case OpType_Cast:
            return tflite::BuiltinOperator_CAST;
        case OpType_ReLU:
            return tflite::BuiltinOperator_RELU;
        case OpType_ReLU6:
            return tflite::BuiltinOperator_RELU6;
        case OpType_Softmax:
            return tflite::BuiltinOperator_SOFTMAX;
        case OpType_Slice:
            return tflite::BuiltinOperator_SPLIT;
        case OpType_Concat:
            return tflite::BuiltinOperator_CONCATENATION;
        case OpType_Reshape:
            return tflite::BuiltinOperator_RESHAPE;
        case OpType_Transpose:
            return tflite::BuiltinOperator_TRANSPOSE;
        case OpType_BinaryOp:
            return getBinaryOpTFLiteType(op->main_as_BinaryOp()->opType());
        case OpType_Eltwise:
            return _getEltwiseOp(op);
        default:
            MNN_PRINT("Warning: Unsupported MNN OpType %d, using CUSTOM operator\n", static_cast<int>(mnnOpType));
            return tflite::BuiltinOperator_CUSTOM;
    }
    return tflite::BuiltinOperator_CUSTOM;
}

ConvertTflite::CommandBuffer SingleConvert::onExecute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) {
    ConvertTflite::CommandBuffer res;
    res.op = op;
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT);
    auto opcode = _mapMNNOpToTFLiteOp(op);
    cmd.op->opcode_index = root->getOpIndex(opcode);
    cmd.inputs = inputs;
    cmd.outputs = outputs;
    if (op->type() == OpType_Softmax) {
        cmd.op->builtin_options.type = tflite::BuiltinOptions_SoftmaxOptions;
        cmd.op->builtin_options.value = new tflite::SoftmaxOptionsT;
        cmd.op->builtin_options.AsSoftmaxOptions()->beta = 1.0f;
    } else if (op->type() == OpType_Slice) {
        cmd.op->builtin_options.type = tflite::BuiltinOptions_SplitOptions;
        cmd.op->builtin_options.value = new tflite::SplitOptionsT;
        cmd.op->builtin_options.AsSplitOptions()->num_splits = (int)cmd.outputs.size();
        std::vector<int> axis = {op->main_as_Slice()->axis()};
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            // Need transform
            auto dimensions = inputs[0]->dimensions();
            if (dimensions >= 3) {
                if (axis[0] == 1) {
                    axis[0] = dimensions - 1;
                } else if (axis[0] > 1) {
                    axis[0] = axis[0] - 1;
                }
            }
        }
        auto axisTensor = ConvertTflite::getIntArrayTensor(axis);
        cmd.inputs = {axisTensor.get(), inputs[0]};
        res.extraConst.emplace_back(axisTensor);
    } else if (op->type() == OpType_Concat) {
        cmd.op->builtin_options.type = tflite::BuiltinOptions_ConcatenationOptions;
        cmd.op->builtin_options.value = new tflite::ConcatenationOptionsT;
        auto tfliteConcatOption = cmd.op->builtin_options.AsConcatenationOptions();
        tfliteConcatOption->axis = op->main_as_Axis()->axis();
    } else if (op->type() == OpType_Cast) {
        cmd.op->builtin_options.type = tflite::BuiltinOptions_CastOptions;
        cmd.op->builtin_options.value = new tflite::CastOptionsT;
        auto cast = cmd.op->builtin_options.AsCastOptions();
        cast->in_data_type = ConvertTflite::getType(inputs[0]);
        cast->out_data_type = ConvertTflite::getType(outputs[0]);
    } else if (op->type() == OpType_StridedSlice) {
        cmd.op->builtin_options.type = tflite::BuiltinOptions_StridedSliceOptions;
        cmd.op->builtin_options.value = new tflite::StridedSliceOptionsT;
        auto src = op->main_as_StridedSliceParam();
        auto dst = cmd.op->builtin_options.AsStridedSliceOptions();
        dst->begin_mask = src->beginMask();
        dst->end_mask = src->endMask();
        dst->ellipsis_mask = src->ellipsisMask();
        dst->new_axis_mask = src->newAxisMask();
        dst->shrink_axis_mask = src->shrinkAxisMask();
        if (src->fromType() == 1) {
            // Change inputs
            std::vector<int> begin(inputs[0]->dimensions(), 0);
            std::vector<int> end(inputs[0]->dimensions());
            for (int i=0; i<inputs[0]->dimensions(); ++i) {
                end[i] = inputs[0]->length(i);
            }
            std::vector<int> stride(inputs[0]->dimensions(), 1);
            auto beginT = inputs[1];
            auto validSize = beginT->elementSize();
            auto endT = inputs[2];
            auto axisT = inputs[3];
            auto stepT = inputs[4];
            for (int i=0; i<validSize; ++i) {
                auto axis = axisT->host<int>()[i];
                if (axis < 0) {
                    axis += inputs[0]->dimensions();
                }
                begin[axis] = beginT->host<int>()[i];
                end[axis] = endT->host<int>()[i];
                stride[axis] = stepT->host<int>()[i];
            }
            auto newBegin = ConvertTflite::getIntArrayTensor(begin);
            auto newEnd = ConvertTflite::getIntArrayTensor(end);
            auto newStride = ConvertTflite::getIntArrayTensor(stride);
            cmd.inputs = {inputs[0], newBegin.get(), newEnd.get(), newStride.get()};
            res.extraConst.emplace_back(newBegin);
            res.extraConst.emplace_back(newEnd);
            res.extraConst.emplace_back(newStride);
        }
    } else if (op->type() == OpType_Reshape) {
        std::vector<int> shape = outputs[0]->shape();
        auto shapeT = ConvertTflite::getIntArrayTensor(shape);
        cmd.inputs = {inputs[0], shapeT.get()};
        res.extraConst.emplace_back(shapeT);
    } else if (op->type() == OpType_Gather || op->type() == OpType_GatherV2) {
        int axis = 0;
        if (inputs.size() == 3) {
            auto axis_tensor = inputs[2];
            axis = axis_tensor->host<int32_t>()[0];
        }
        if (op->main_type() == OpParameter_Axis) {
            axis = op->main_as_Axis()->axis();
        }
        if (inputs[0]->size() == outputs[0]->size()) {
            cmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_RESHAPE);
            std::vector<int> shape = outputs[0]->shape();
            auto shapeT = ConvertTflite::getIntArrayTensor(shape);
            cmd.inputs = {inputs[0], shapeT.get()};
            res.extraConst.emplace_back(shapeT);
        } else if(TensorUtils::getDescribe(inputs[1])->usage == Tensor::InsideDescribe::CONSTANT && inputs[1]->elementSize() == 1) {
            // Turn to Slice
            cmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_STRIDED_SLICE);
            cmd.op->builtin_options.type = tflite::BuiltinOptions_StridedSliceOptions;
            cmd.op->builtin_options.value = new tflite::StridedSliceOptionsT;
            std::vector<int> begin(inputs[0]->dimensions(), 0);
            begin[axis] = inputs[1]->host<int>()[0];
            std::vector<int> end(inputs[0]->dimensions());
            for (int i=0; i<inputs[0]->dimensions(); ++i) {
                end[i] = inputs[0]->length(i);
            }
            end[axis] = begin[axis] + 1;
            cmd.op->builtin_options.AsStridedSliceOptions()->shrink_axis_mask =  1 << axis;
            std::vector<int> stride(inputs[0]->dimensions(), 1);
            auto newBegin = ConvertTflite::getIntArrayTensor(begin);
            auto newEnd = ConvertTflite::getIntArrayTensor(end);
            auto newStride = ConvertTflite::getIntArrayTensor(stride);
            cmd.inputs = {inputs[0], newBegin.get(), newEnd.get(), newStride.get()};
            res.extraConst.emplace_back(newBegin);
            res.extraConst.emplace_back(newEnd);
            res.extraConst.emplace_back(newStride);
        } else {
            cmd.op->builtin_options.type = tflite::BuiltinOptions_GatherOptions;
            cmd.op->builtin_options.value = new tflite::GatherOptionsT;
            auto dst = cmd.op->builtin_options.AsGatherOptions();
            dst->axis = axis;
            cmd.inputs = {inputs[0], inputs[1]};
        }
    }
    res.commands.emplace_back(std::move(cmd));
    return res;
}

};
