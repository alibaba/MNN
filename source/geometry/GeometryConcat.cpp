//
//  GeometryConcat.cpp
//  MNN
//
//  Created by MNN on 2020/04/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
namespace MNN {
class GeometryConcat : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(inputs.size() >= 1);
        int basicAxis = 0;
        bool inputFix = false;
        if (op->type() == OpType_Concat) {
            basicAxis = op->main_as_Axis()->axis();
        } else if (op->type() == OpType_QuantizedConcat) {
            basicAxis = op->main_as_QuantizedConcat()->axis();
        } else if (op->type() == OpType_Pack) {
            basicAxis = op->main_as_PackParam()->axis();
            inputFix  = true;
        }
        auto output = outputs[0];
        int axis    = basicAxis;
        if (axis < 0) {
            axis = output->dimensions() + axis;
        }
        auto outputDes        = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

        int outside = 1;
        int inside  = 1;
        for (int i = 0; i < axis; ++i) {
            outside *= output->length(i);
        }
        for (int i = axis + 1; i < output->dimensions(); ++i) {
            inside *= output->length(i);
        }
        int offset = 0;
        outputDes->regions.clear();
        outputDes->regions.reserve(inputs.size());
        auto axisLength = output->length(axis);
        if (outside <= 0 || inside <= 0 || axisLength <= 0) {
            return true;
        }

        for (int i = 0; i < inputs.size(); ++i) {
            auto t = inputs[i];
            if (t->elementSize() == 0) {
                continue;
            }
            Tensor::InsideDescribe::Region dstSlice;
            int basicStride = 1;
            int basicOffset = 0;

            dstSlice.origin     = t;
            dstSlice.src.offset = basicOffset;
            dstSlice.dst.offset = offset * inside;
            dstSlice.size[0]    = outside;
            if (inputFix) {
                dstSlice.size[1] = 1;
            } else {
                dstSlice.size[1] = t->length(axis);
            }
            dstSlice.dst.stride[0] = inside * axisLength;
            dstSlice.dst.stride[1] = inside;
            dstSlice.dst.stride[2] = 1;
            dstSlice.size[2]       = inside;
            offset += dstSlice.size[1];
            dstSlice.src.stride[0] = basicStride * dstSlice.size[1] * dstSlice.size[2];
            dstSlice.src.stride[1] = basicStride * dstSlice.size[2];
            dstSlice.src.stride[2] = basicStride;
            outputDes->regions.emplace_back(std::move(dstSlice));
        }
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryConcat);
    GeometryComputer::registerGeometryComputer(comp, {OpType_QuantizedConcat, OpType_Concat, OpType_Pack});
}

REGISTER_GEOMETRY(GeometryConcat, _create);

} // namespace MNN
