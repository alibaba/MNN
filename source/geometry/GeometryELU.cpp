//
//  GeometryELU.cpp
//  MNN
//
//  Created by MNN on 2020/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {

static void initTensor(std::shared_ptr<Tensor> tensor, Tensor* input) {
    TensorUtils::copyShape(input, tensor.get(), true, true);
}

class GeometryELU : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto input = inputs[0];
        auto output = outputs[0];
        // ELU : y = x > 0 ? x : alpha * (exp(x) - 1)
        // exp + sub + mul : y1 = alhpa * (exp(x) - 1)
        // exp
        std::shared_ptr<Tensor> expValue(new Tensor);
        {
            initTensor(expValue, input);
            auto cmd = GeometryComputerUtils::makeUnary(UnaryOpOperation_EXP, input, expValue.get());
            res.extras.emplace_back(expValue);
            res.command.emplace_back(std::move(cmd));
        }
        // sub
        std::shared_ptr<Tensor> subValue(new Tensor);
        {
            auto oneConst = context.allocConst(op, {}, halide_type_of<float>());
            oneConst->host<float>()[0] = 1.0;
            initTensor(subValue, input);
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_SUB, expValue.get(), oneConst.get(), subValue.get());
            res.extras.emplace_back(subValue);
            res.command.emplace_back(std::move(cmd));
        }
        // mul
        std::shared_ptr<Tensor> mulValue(new Tensor);
        {
            auto alphaConst = context.allocConst(op, {}, halide_type_of<float>());
            float alpha = 0.0;
            if (op->type() == OpType_ELU) {
                alpha = op->main_as_ELU()->alpha();
            } else if (op->type() == OpType_Selu){
                alpha = op->main_as_Selu()->alpha() *
                        op->main_as_Selu()->scale();
            }
            alphaConst->host<float>()[0] = alpha;
            initTensor(mulValue, input);
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, subValue.get(), alphaConst.get(), mulValue.get());
            res.extras.emplace_back(mulValue);
            res.command.emplace_back(std::move(cmd));
        }
        // compare + select : y = x > 0 ? x : y1
        // compare
        std::shared_ptr<Tensor> compValue(new Tensor);
        {
            auto zeroConst = context.allocConst(op, {}, halide_type_of<float>());
            zeroConst->host<float>()[0] = 0;
            TensorUtils::copyShape(input, compValue.get(), true, true);
            compValue->buffer().type = halide_type_of<int32_t>();
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_GREATER, input, zeroConst.get(), compValue.get());
            res.extras.emplace_back(compValue);
            res.command.emplace_back(std::move(cmd));
        }
        std::shared_ptr<Tensor> scaleValue(new Tensor);
        {
            if (op->type() == OpType_Selu) {
                auto scaleConst = context.allocConst(op, {}, halide_type_of<float>());
                float scale = op->main_as_Selu()->scale();
                scaleConst->host<float>()[0] = scale;
                initTensor(scaleValue, input);
                auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, input, scaleConst.get(), scaleValue.get());
                res.extras.emplace_back(scaleValue);
                res.command.emplace_back(std::move(cmd));
            }
        }
        // select
        {
            flatbuffers::FlatBufferBuilder builder;
            OpBuilder opBuilder(builder);
            opBuilder.add_type(OpType_Select);
            builder.Finish(opBuilder.Finish());
            auto y0 = op->type() == OpType_ELU ? input : scaleValue.get();
            auto cmd = GeometryComputerUtils::makeCommand(builder, {compValue.get(), y0, mulValue.get()}, {output});
            res.command.emplace_back(std::move(cmd));
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryELU);
    GeometryComputer::registerGeometryComputer(comp, {OpType_ELU, OpType_Selu});
}

REGISTER_GEOMETRY(GeometryELU, _create);

} // namespace MNN

