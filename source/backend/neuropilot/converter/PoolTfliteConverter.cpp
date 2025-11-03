#include "PoolTfliteConverter.hpp"
namespace MNN {
ConvertTflite::CommandBuffer PoolTfliteConverter::onExecute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) {
    ConvertTflite::CommandBuffer res;
    res.op = op;
    ConvertTflite::Command cmd;
    cmd.op.reset(new tflite::OperatorT());
    auto pool = op->main_as_Pool();
    auto type = pool->type();
    if (type == PoolType_AVEPOOL) {
        cmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_AVERAGE_POOL_2D);
    }
    if (type == PoolType_MAXPOOL) {
        cmd.op->opcode_index = root->getOpIndex(tflite::BuiltinOperator_MAX_POOL_2D);
    }
    cmd.op->builtin_options.type = tflite::BuiltinOptions_Pool2DOptions;
    cmd.op->builtin_options.value = new tflite::Pool2DOptionsT;
    if (pool->isGlobal()) {
        cmd.op->builtin_options.AsPool2DOptions()->stride_h = 1;
        cmd.op->builtin_options.AsPool2DOptions()->stride_w = 1;
        cmd.op->builtin_options.AsPool2DOptions()->filter_height = inputs[0]->height();
        cmd.op->builtin_options.AsPool2DOptions()->filter_width = inputs[0]->width();
    } else {
        cmd.op->builtin_options.AsPool2DOptions()->stride_h = pool->strideY();
        cmd.op->builtin_options.AsPool2DOptions()->stride_w = pool->strideX();
        cmd.op->builtin_options.AsPool2DOptions()->filter_height = pool->kernelY();
        cmd.op->builtin_options.AsPool2DOptions()->filter_width = pool->kernelX();
    }
    // TODO: Add extra padding for pads
    cmd.op->builtin_options.AsPool2DOptions()->padding = tflite::Padding_SAME;
    switch (pool->padType()) {
        case PoolPadType_VALID:
            cmd.op->builtin_options.AsPool2DOptions()->padding = tflite::Padding_VALID;
            break;
        default:
            break;
    }
    if (pool->isGlobal()) {
        cmd.op->builtin_options.AsPool2DOptions()->padding = tflite::Padding_VALID;
    }
    cmd.inputs = inputs;
    cmd.outputs = outputs;
    res.commands.emplace_back(std::move(cmd));
    return res;
}

};
