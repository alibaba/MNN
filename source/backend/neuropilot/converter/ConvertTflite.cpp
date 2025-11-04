#include "core/TensorUtils.hpp"
#include "shape/SizeComputer.hpp"
#include "ConvertTflite.hpp"
#include "ConvolutionTfliteConverter.hpp"
#include "PoolTfliteConverter.hpp"
#include "SingleConvert.hpp"
#include "UnaryTfliteConverter.hpp"
#include "AttentionConverter.hpp"
#include "MTKEXT.hpp"
namespace MNN {
std::shared_ptr<Tensor> ConvertTflite::getIntArrayTensor(std::vector<int> shapes) {
    std::shared_ptr<Tensor> shape(Tensor::create<int>({(int)shapes.size()}));
    ::memcpy(shape->host<int>(), shapes.data(), shapes.size() * sizeof(int));
    TensorUtils::getDescribe(shape.get())->usage = Tensor::InsideDescribe::CONSTANT;
    return shape;
}
Tensor* ConvertTflite::makeSlice(CommandBuffer& res, Tensor* input, int sta, int size, int axis) {
    if (axis < 0) {
        axis = axis + input->dimensions();
    }
    auto shape = input->shape();
    shape[axis] = size;
    std::shared_ptr<Tensor> output(Tensor::createDevice(shape, input->getType()));
    TensorUtils::getDescribe(output.get())->applyQuant = TensorUtils::getDescribe(input)->applyQuant;
    TensorUtils::getDescribe(output.get())->quantAttr = TensorUtils::getDescribe(input)->quantAttr;
    res.extraConst.emplace_back(output);
    std::vector<int> begin(shape.size(), 0);
    begin[axis] = sta;
    std::vector<int> end = shape;
    end[axis] = sta + size;
    std::vector<int> stride(shape.size(), 1);
    auto beginT = getIntArrayTensor(begin);
    auto endT = getIntArrayTensor(end);
    auto strideT = getIntArrayTensor(stride);
    res.extraConst.emplace_back(beginT);
    res.extraConst.emplace_back(endT);
    res.extraConst.emplace_back(strideT);

    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT());
    cmd.op->opcode_index = getOpIndex(tflite::BuiltinOperator_STRIDED_SLICE);
    cmd.op->builtin_options.type = tflite::BuiltinOptions_StridedSliceOptions;
    cmd.op->builtin_options.value = new tflite::StridedSliceOptionsT;
    cmd.inputs = {input, beginT.get(), endT.get(), strideT.get()};
    cmd.outputs = {output.get()};
    res.commands.emplace_back(std::move(cmd));
    return output.get();
}

Tensor* ConvertTflite::makeConcat(CommandBuffer& res, std::vector<Tensor*> inputs, int axis) {
    auto i0 = inputs[0];
    if (axis < 0) {
        axis = axis + i0->dimensions();
    }
    auto shape = i0->shape();
    for (int i=1; i<inputs.size(); ++i) {
        shape[axis] += inputs[i]->length(axis);
    }
    std::shared_ptr<Tensor> output(Tensor::createDevice(shape, i0->getType()));
    res.extraConst.emplace_back(output);
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT());
    cmd.op->opcode_index = getOpIndex(tflite::BuiltinOperator_CONCATENATION);
    cmd.op->builtin_options.type = tflite::BuiltinOptions_ConcatenationOptions;
    cmd.op->builtin_options.value = new tflite::ConcatenationOptionsT;
    cmd.op->builtin_options.AsConcatenationOptions()->axis = axis;
    cmd.inputs = inputs;
    cmd.outputs = {output.get()};
    res.commands.emplace_back(std::move(cmd));
    return output.get();
}

Tensor* ConvertTflite::makeReshape(CommandBuffer& res, Tensor* tensor, std::vector<int> reshapeSize, Tensor* outputUser) {
    ConvertTflite::Command cmd;
    auto reshapeTensor = ConvertTflite::getIntArrayTensor(reshapeSize);
    res.extraConst.emplace_back(reshapeTensor);
    if (outputUser == nullptr) {
        std::shared_ptr<Tensor> reshapeOutput(Tensor::createDevice(reshapeSize, tensor->getType()));
        TensorUtils::getDescribe(reshapeOutput.get())->applyQuant = TensorUtils::getDescribe(tensor)->applyQuant;
        TensorUtils::getDescribe(reshapeOutput.get())->quantAttr = TensorUtils::getDescribe(tensor)->quantAttr;
        res.extraConst.emplace_back(reshapeOutput);
        outputUser = reshapeOutput.get();
    }
    cmd.op.reset(new tflite::OperatorT());
    cmd.op->opcode_index = getOpIndex(tflite::BuiltinOperator_RESHAPE);
    cmd.outputs = {outputUser};
    cmd.inputs = {tensor, reshapeTensor.get()};
    res.commands.emplace_back(std::move(cmd));
    return outputUser;
}
Tensor* ConvertTflite::makeTranspose(CommandBuffer& res, Tensor* tensor, std::vector<int> dims) {
    auto dimTensor = ConvertTflite::getIntArrayTensor(dims);
    auto tensorShape = getShapeOfTensor(tensor);
    std::vector<int> reshapeSize(tensorShape.size());
    MNN_ASSERT(dims.size() == tensorShape.size());
    for (int i=0; i<dims.size(); ++i) {
        reshapeSize[i] = tensorShape[dims[i]];
    }
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT());
    cmd.op->opcode_index = getOpIndex(tflite::BuiltinOperator_TRANSPOSE);
    std::shared_ptr<Tensor> reshapeOutput(Tensor::createDevice(reshapeSize, tensor->getType()));
    TensorUtils::getDescribe(reshapeOutput.get())->applyQuant = TensorUtils::getDescribe(tensor)->applyQuant;
    TensorUtils::getDescribe(reshapeOutput.get())->quantAttr = TensorUtils::getDescribe(tensor)->quantAttr;
    cmd.outputs = {reshapeOutput.get()};
    cmd.inputs = {tensor, dimTensor.get()};
    auto output = reshapeOutput.get();
    res.extraConst.emplace_back(dimTensor);
    res.extraConst.emplace_back(reshapeOutput);
    res.commands.emplace_back(std::move(cmd));
    return output;
}
Tensor* ConvertTflite::makeTile(CommandBuffer& res, Tensor* tensor, std::vector<int> dims) {
    auto dimTensor = ConvertTflite::getIntArrayTensor(dims);
    auto tensorShape = getShapeOfTensor(tensor);
    std::vector<int> reshapeSize(tensorShape.size());
    MNN_ASSERT(dims.size() == tensorShape.size());
    for (int i=0; i<dims.size(); ++i) {
        reshapeSize[i] = tensorShape[i] * dims[i];
    }
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT());
    cmd.op->opcode_index = getOpIndex(tflite::BuiltinOperator_TILE);
    std::shared_ptr<Tensor> reshapeOutput(Tensor::createDevice(reshapeSize, tensor->getType()));
    TensorUtils::getDescribe(reshapeOutput.get())->applyQuant = TensorUtils::getDescribe(tensor)->applyQuant;
    TensorUtils::getDescribe(reshapeOutput.get())->quantAttr = TensorUtils::getDescribe(tensor)->quantAttr;
    cmd.outputs = {reshapeOutput.get()};
    cmd.inputs = {tensor, dimTensor.get()};
    auto output = reshapeOutput.get();
    res.extraConst.emplace_back(dimTensor);
    res.extraConst.emplace_back(reshapeOutput);
    res.commands.emplace_back(std::move(cmd));
    return output;
}
Tensor* ConvertTflite::makeBinary(CommandBuffer& res, Tensor* A, Tensor* B, tflite::BuiltinOperator operation) {
    std::shared_ptr<Tensor> tensor(Tensor::createDevice({}, A->getType()));
    SizeComputer::computeBroadCastDims({A, B}, {tensor.get()});
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT());
    cmd.op->opcode_index = getOpIndex(operation);
    cmd.inputs = {A, B};
    cmd.outputs = {tensor.get()};
    res.extraConst.emplace_back(tensor);
    res.commands.emplace_back(std::move(cmd));
    return tensor.get();
}
void ConvertTflite::makeMatMul(CommandBuffer& res, Tensor* A, Tensor* B, bool adjA, bool adjB, Tensor* C) {
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT());
    cmd.op->opcode_index = getOpIndex(tflite::BuiltinOperator_BATCH_MATMUL);
    cmd.op->builtin_options.type = tflite::BuiltinOptions_BatchMatMulOptions;
    cmd.op->builtin_options.value = new tflite::BatchMatMulOptionsT;
    cmd.op->builtin_options.AsBatchMatMulOptions()->adj_x = adjA;
    cmd.op->builtin_options.AsBatchMatMulOptions()->adj_y = adjB;
    cmd.inputs = {A, B};
    cmd.outputs = {C};
    res.commands.emplace_back(std::move(cmd));
}

Tensor* ConvertTflite::makeSoftmax(CommandBuffer& res, Tensor* A) {
    std::shared_ptr<Tensor> tensor(Tensor::createDevice(getShapeOfTensor(A), A->getType()));
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT());
    cmd.op->opcode_index = getOpIndex(tflite::BuiltinOperator_SOFTMAX);
    cmd.op->builtin_options.type = tflite::BuiltinOptions_SoftmaxOptions;
    cmd.op->builtin_options.value = new tflite::SoftmaxOptionsT;
    cmd.op->builtin_options.AsSoftmaxOptions()->beta = 1.0f;
    cmd.inputs = {A};
    cmd.outputs = {tensor.get()};
    res.extraConst.emplace_back(tensor);
    res.commands.emplace_back(std::move(cmd));
    return tensor.get();
}


class ConvertTensorTflite : public ConvertTflite::Convert {
public:
    virtual ConvertTflite::CommandBuffer onExecute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) override {
        auto srcFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        auto dstFormat = TensorUtils::getDescribe(outputs[0])->dimensionFormat;
        if (MNN_DATA_FORMAT_NC4HW4 == srcFormat) {
            srcFormat = MNN_DATA_FORMAT_NHWC;
        }
        if (MNN_DATA_FORMAT_NC4HW4 == dstFormat) {
            dstFormat = MNN_DATA_FORMAT_NHWC;
        }
        ConvertTflite::CommandBuffer res;
        res.op = op;
        ConvertTflite::Command cmd;
        cmd.op.reset(new tflite::OperatorT());
        auto batchChannel = inputs[0]->batch() * inputs[0]->channel();
        if (srcFormat == dstFormat || inputs[0]->dimensions() == 2 || batchChannel == inputs[0]->elementSize()) {
            // Reshape
            cmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_RESHAPE);
            auto shapes = ConvertTflite::getShapeOfTensor(outputs[0]);
            auto shape = ConvertTflite::getIntArrayTensor(shapes);
            cmd.inputs = {inputs[0], shape.get()};
            cmd.outputs = outputs;
            res.extraConst.emplace_back(std::move(shape));
        } else {
            cmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_TRANSPOSE);
            std::vector<int> permutes(inputs[0]->dimensions());
            if (srcFormat == MNN_DATA_FORMAT_NHWC) {
                permutes[0] = 0;
                permutes[1] = inputs[0]->dimensions() - 1;
                for (int i=1; i<inputs[0]->dimensions()-1; ++i) {
                    permutes[i+1] = i;
                }
            } else {
                // NCHW -> NHWC
                permutes[0] = 0;
                permutes[inputs[0]->dimensions()-1] = 1;
                for (int i=1; i<inputs[0]->dimensions()-1; ++i) {
                    permutes[i] = i + 1;
                }
            }
            auto shape = ConvertTflite::getIntArrayTensor(permutes);
            cmd.inputs = {inputs[0], shape.get()};
            cmd.outputs = outputs;
            res.extraConst.emplace_back(std::move(shape));
        }
        res.commands.emplace_back(std::move(cmd));
        return res;
    }

};
ConvertTflite::ConvertTflite() {
    {
        std::shared_ptr<Convert> single(new SingleConvert);
        mConverters.insert(std::make_pair(OpType_Concat, single));
        mConverters.insert(std::make_pair(OpType_ReLU, single));
        mConverters.insert(std::make_pair(OpType_ReLU6, single));
        mConverters.insert(std::make_pair(OpType_Reshape, single));
        mConverters.insert(std::make_pair(OpType_Transpose, single));
        mConverters.insert(std::make_pair(OpType_Softmax, single));
        mConverters.insert(std::make_pair(OpType_BinaryOp, single));
        mConverters.insert(std::make_pair(OpType_Eltwise, single));
        mConverters.insert(std::make_pair(OpType_Cast, single));
        mConverters.insert(std::make_pair(OpType_StridedSlice, single));
        mConverters.insert(std::make_pair(OpType_Slice, single));
        mConverters.insert(std::make_pair(OpType_GatherV2, single));
        mConverters.insert(std::make_pair(OpType_Gather, single));
        mConverters.insert(std::make_pair(OpType_Int8ToFloat, single));
        mConverters.insert(std::make_pair(OpType_FloatToInt8, single));
    }
    {
        std::shared_ptr<Convert> conv(new ConvolutionTfliteConverter);
        mConverters.insert(std::make_pair(OpType_Convolution, conv));
        mConverters.insert(std::make_pair(OpType_ConvolutionDepthwise, conv));
    }
    {
        std::shared_ptr<Convert> attn(new AttentionConverter);
        mConverters.insert(std::make_pair(OpType_Attention, attn));
    }
    {
        std::shared_ptr<Convert> pool(new PoolTfliteConverter);
        mConverters.insert(std::make_pair(OpType_Pooling, pool));
    }
    {
        std::shared_ptr<Convert> pool(new UnaryTfliteConverter);
        mConverters.insert(std::make_pair(OpType_UnaryOp, pool));
    }
    {
        std::shared_ptr<Convert> convert(new ConvertTensorTflite);
        mConverters.insert(std::make_pair(OpType_ConvertTensor, convert));
        mConverters.insert(std::make_pair(OpType_Squeeze, convert));
        mConverters.insert(std::make_pair(OpType_Unsqueeze, convert));
        mConverters.insert(std::make_pair(OpType_Flatten, convert));
        mConverters.insert(std::make_pair(OpType_Identity, convert));
    }
    {
        std::shared_ptr<Convert> convert(new MTKEXT);
        mConverters.insert(std::make_pair(OpType_LayerNorm, convert));
    }
}
ConvertTflite::~ ConvertTflite() {
    // Do nothing
}

int ConvertTflite::getOpIndex(tflite::BuiltinOperator op) {
    if (mOperatorCodeIndexMap.find(op) != mOperatorCodeIndexMap.end()) {
        return mOperatorCodeIndexMap.find(op)->second;
    }
    int res = (int)mOperatorCodes.size();
    mOperatorCodeIndexMap.insert(std::make_pair(op, res));
    std::unique_ptr<tflite::OperatorCodeT> operatorCode(new tflite::OperatorCodeT());
    operatorCode->builtin_code = op;
    operatorCode->version = 1;
    mOperatorCodes.emplace_back(std::move(operatorCode));
    return res;
}
int ConvertTflite::getCustomOpIndex(std::string name) {
    auto iter = mCustomOpIndex.find(name);
    if (iter != mCustomOpIndex.end()) {
        return iter->second;
    }
    int res = (int)mOperatorCodes.size();
    std::unique_ptr<tflite::OperatorCodeT> operatorCode(new tflite::OperatorCodeT());
    operatorCode->builtin_code = tflite::BuiltinOperator_CUSTOM;
    operatorCode->custom_code = name;
    operatorCode->version = 1;
    mCustomOpIndex.insert(std::make_pair(name, res));
    mOperatorCodes.emplace_back(std::move(operatorCode));
    return res;
}

ConvertTflite::CommandBuffer ConvertTflite::convert(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto iter = mConverters.find(op->type());
    if (iter != mConverters.end()) {
        return iter->second->onExecute(op, inputs, outputs, this);
    }
    MNN_ERROR("Don't support op convert: %s\n", EnumNameOpType(op->type()));
    CommandBuffer res;
    res.op = op;
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT);
    cmd.op->opcode_index = getCustomOpIndex("Unknown");
    cmd.inputs = inputs;
    cmd.outputs = outputs;
    res.commands.emplace_back(std::move(cmd));

    return res;
}

std::vector<int> ConvertTflite::getShapeOfTensor(const Tensor* tensor) {
    auto des = TensorUtils::getDescribe(tensor);
    std::vector<int> shape;
    if (des->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && tensor->dimensions() > 2) {
        // Use NHWC instead of NC4HW4
        shape.emplace_back(tensor->length(0));
        for (int i=2; i<tensor->dimensions(); ++i) {
            shape.emplace_back(tensor->length(i));
        }
        shape.emplace_back(tensor->length(1));
    } else {
        for (int i = 0; i < tensor->dimensions(); ++i) {
            shape.push_back(tensor->length(i));
        }
    }
    return shape;
}

tflite::TensorType ConvertTflite::getType(const Tensor* tensor) {
    auto des = TensorUtils::getDescribe(tensor);
    if (des->applyQuant && des->quantAttr.get() != nullptr) {
        if (DataType_DT_INT8 == des->quantAttr->type) {
            return tflite::TensorType_INT8;
        } else if (DataType_DT_INT16 == des->quantAttr->type) {
            return tflite::TensorType_INT16;
        }
        MNN_ERROR("ConvertTflite Don't support quant type: %d\n", des->quantAttr->type);
        return tflite::TensorType_FLOAT32;
    }
    tflite::TensorType type;
    switch (tensor->getType().code) {
        case halide_type_float:
            type = tflite::TensorType_FLOAT32;
            break;
        case halide_type_int:
            if (tensor->getType().bits == 32) {
                type = tflite::TensorType_INT32;
            } else if (tensor->getType().bits == 8) {
                type = tflite::TensorType_INT8;
            }
            break;
        case halide_type_uint:
            if (tensor->getType().bits == 8) {
                type = tflite::TensorType_UINT8;
            }
            break;
        default:
            type = tflite::TensorType_FLOAT32; // 默认
            break;
    }
    return type;
}
};
