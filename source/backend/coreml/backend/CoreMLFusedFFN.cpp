// Copyright © Alibaba Group Holding Limited. All rights reserved.
#include "CoreMLFusedFFN.hpp"
#include "CoreMLHelper.hpp"
#include <climits>

namespace MNN {
bool CoreMLFusedFFN::prepare(int tokens, int hidden, int intermediate, const std::string& weightsPath,
                             const std::vector<uint64_t>& offsets, bool fp32Gate) {
    if (tokens < 1 || hidden < 1 || intermediate < 1 || weightsPath.empty() || offsets.size() != 3 ||
        int64_t(hidden) * intermediate > INT_MAX || int64_t(tokens) * intermediate > INT_MAX ||
        int64_t(tokens) * hidden > INT_MAX) {
        return false;
    }
    const auto fp16 = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16;
    const auto fp32 = CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32;
    MlModelBuilder builder(9);
    auto io = [&](const std::string& name, bool input) {
        std::unique_ptr<MilTensorInfo> info(new MilTensorInfo);
        info->name = name;
        info->data_type = fp16;
        info->shape = {tokens, hidden};
        info->type = input ? MilTensorInfo::Type::INPUT : MilTensorInfo::Type::OUTPUT;
        if (input) {
            builder.addInput(std::move(info));
        } else {
            builder.addOutput(std::move(info));
        }
    };
    io("input", true);
    io("output", false);
    auto scalar = [&](const std::string& name, std::unique_ptr<MilValueBuilder> value,
                      CoreML__Specification__MILSpec__DataType type) {
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("const"));
        op->setName(name);
        op->addOutput(name, type, {});
        op->addAttribute("val", std::move(value));
        builder.addOperation(std::move(op));
    };
    scalar("no_transpose", MilValueBuilder::createMilValueBuilder(false),
           CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL);
    scalar("transpose", MilValueBuilder::createMilValueBuilder(true), CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__BOOL);
    const char* weights[] = {"gate_weight", "up_weight", "down_weight"};
    for (int i = 0; i < 3; ++i) {
        std::vector<int> shape =
            i == 2 ? std::vector<int>{hidden, intermediate} : std::vector<int>{intermediate, hidden};
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("const"));
        op->setName(weights[i]);
        op->addOutput(weights[i], fp16, {shape[0], shape[1]});
        op->addAttribute("val", MilValueBuilder::createMilValueBuilder(std::string("@model_path/weights/weight.bin"),
                                                                       fp16, shape, offsets[i]));
        builder.addOperation(std::move(op));
    }
    auto matmul = [&](const std::string& x, const std::string& weight, const std::string& out, int width) {
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder("matmul"));
        op->setName(out);
        op->addInput("x", x);
        op->addInput("y", weight);
        op->addInput("transpose_x", "no_transpose");
        op->addInput("transpose_y", "transpose");
        op->addOutput(out, fp16, {tokens, width});
        builder.addOperation(std::move(op));
    };
    auto elementwise = [&](const char* type, const std::string& x, const std::string& y, const std::string& out,
                           CoreML__Specification__MILSpec__DataType dtype) {
        std::unique_ptr<MilOperationBuilder> op(new MilOperationBuilder(type));
        op->setName(out);
        op->addInput("x", x);
        if (!y.empty()) {
            op->addInput(std::string(type) == "cast" ? "dtype" : "y", y);
        }
        op->addOutput(out, dtype, {tokens, intermediate});
        builder.addOperation(std::move(op));
    };
    matmul("input", weights[0], "gate", intermediate);
    matmul("input", weights[1], "up", intermediate);
    std::string gate = "gate", up = "up";
    if (fp32Gate) {
        scalar("float_type", MilValueBuilder::createMilValueBuilder(std::string("fp32")),
               CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__STRING);
        scalar("half_type", MilValueBuilder::createMilValueBuilder(std::string("fp16")),
               CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__STRING);
        elementwise("cast", gate, "float_type", "gate_float", fp32);
        elementwise("cast", up, "float_type", "up_float", fp32);
        gate = "gate_float";
        up = "up_float";
    }
    const auto dtype = fp32Gate ? fp32 : fp16;
    elementwise("sigmoid", gate, "", "sigmoid", dtype);
    elementwise("mul", gate, "sigmoid", "silu", dtype);
    elementwise("mul", up, "silu", "gated", dtype);
    if (fp32Gate) {
        elementwise("cast", "gated", "half_type", "gated_half", fp16);
    }
    matmul(fp32Gate ? "gated_half" : "gated", weights[2], "output", hidden);
    std::shared_ptr<CoreMLExecutorWrapper> program(new CoreMLExecutorWrapper(0));
    if (!program->compileModel(builder.build(), weightsPath)) {
        return false;
    }
    mProgram = std::move(program);
    mTokens = tokens;
    mHidden = hidden;
    return true;
}

bool CoreMLFusedFFN::compiledFromSource() const {
    return mProgram && mProgram->compiledFromSource();
}

bool CoreMLFusedFFN::run(const Tensor* input, const Tensor* output) {
    if (!mProgram || !input || !output || !input->host<void>() || !output->host<void>()) {
        return false;
    }
    for (auto tensor : {input, output}) {
        if (tensor->dimensions() != 2 || tensor->length(0) != mTokens || tensor->length(1) != mHidden ||
            tensor->getType() != halide_type_t(halide_type_float, 16) || tensor->stride(1) != 1 ||
            tensor->stride(0) != mHidden) {
            return false;
        }
    }
    return mProgram->invokModel({{input, "input"}}, {{output, "output"}});
}
} // namespace MNN
