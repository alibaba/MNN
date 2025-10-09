#include "UnaryTfliteConverter.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
static tflite::BuiltinOperator _mapUnary(UnaryOpOperation src) {
    switch (src) {
        case UnaryOpOperation_ABS:
            return tflite::BuiltinOperator_ABS;
        case UnaryOpOperation_COS:
            return tflite::BuiltinOperator_COS;
        case UnaryOpOperation_EXP:
            return tflite::BuiltinOperator_EXP;
        case UnaryOpOperation_LOG:
            return tflite::BuiltinOperator_LOG;
        case UnaryOpOperation_NEG:
            return tflite::BuiltinOperator_NEG;
        case UnaryOpOperation_ROUND:
            return tflite::BuiltinOperator_ROUND;
        case UnaryOpOperation_SIN:
            return tflite::BuiltinOperator_SIN;
        case UnaryOpOperation_SIGMOID:
            return tflite::BuiltinOperator_LOGISTIC;
        case UnaryOpOperation_SQRT:
            return tflite::BuiltinOperator_SQRT;
        case UnaryOpOperation_SQUARE:
            return tflite::BuiltinOperator_SQUARE;
        case UnaryOpOperation_TANH:
            return tflite::BuiltinOperator_TANH;
        default:
            break;
    }
    return tflite::BuiltinOperator_CUSTOM;
}
ConvertTflite::CommandBuffer UnaryTfliteConverter::onExecute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) {
    ConvertTflite::CommandBuffer res;
    res.op = op;
    auto param = op->main_as_UnaryOp();
    auto code = _mapUnary(param->opType());
    if (code != tflite::BuiltinOperator_CUSTOM) {
        ConvertTflite::Command cmd;
        cmd.op.reset(new tflite::OperatorT());
        cmd.op->opcode_index = root->getOpIndex(code);
        cmd.inputs = inputs;
        cmd.outputs = outputs;
        res.commands.emplace_back(std::move(cmd));
        return res;
    }
    if (param->opType() == UnaryOpOperation_SILU) {
        // sigmoid + mul
        std::shared_ptr<Tensor> tensor(new Tensor(inputs[0], inputs[0]->getDimensionType(), false));
        TensorUtils::getDescribe(tensor.get())->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        // Sigmoid
        ConvertTflite::Command sigmoidcmd;
        sigmoidcmd.op.reset(new tflite::OperatorT());
        sigmoidcmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_LOGISTIC);
        sigmoidcmd.inputs = inputs;
        sigmoidcmd.outputs = {tensor.get()};
        res.commands.emplace_back(std::move(sigmoidcmd));
        // Mul
        ConvertTflite::Command mulcmd;
        mulcmd.op.reset(new tflite::OperatorT());
        mulcmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_MUL);
        mulcmd.inputs = {inputs[0], tensor.get()};
        mulcmd.outputs = outputs;
        res.commands.emplace_back(std::move(mulcmd));

        res.extraConst.emplace_back(tensor);
        return res;
    }
    MNN_ERROR("Don't support convert %s unary to tflite\n", EnumNameUnaryOpOperation(param->opType()));
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT());
    cmd.op->opcode_index = root->getCustomOpIndex("UNKNOWN");
    cmd.inputs = inputs;
    cmd.outputs = outputs;
    res.commands.emplace_back(std::move(cmd));
    return res;
}

};
