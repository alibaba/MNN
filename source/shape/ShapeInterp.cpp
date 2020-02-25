//
//  ShapeInterp.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/SizeComputer.hpp"

namespace MNN {

// Size Computer
class InterpComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto& input         = inputs[0]->buffer(); // input tensor(data)
        auto& output        = outputs[0]->buffer();
        int w               = 0;
        int h               = 0;
        const int inputSize = (int)inputs.size();
        // copy dims
        memcpy(output.dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);
        if (1 == inputSize) {
            auto interp = op->main_as_Interp();
            // get output dims
            w = interp->outputWidth();
            h = interp->outputHeight();
            if (w == 0 || h == 0) {
                w = input.dim[3].extent * interp->widthScale();
                h = input.dim[2].extent * interp->heightScale();
            }
            output.dim[3].extent = w;
            output.dim[2].extent = h;
        } else {
            // copy data from device to host if needed
            std::shared_ptr<Tensor> tmpShape;
            auto shape = inputs[1]; // input shape(shape)
            if (!shape->host<int32_t>() && shape->deviceId()) {
                tmpShape.reset(Tensor::createHostTensorFromDevice(shape, true));
                shape = tmpShape.get();
            }

            MNN_ASSERT(2 == shape->buffer().dim[0].extent);
            const int32_t* shapeData = shape->host<int32_t>();
            w                        = shapeData[1];
            h                        = shapeData[0];
            output.dim[3].extent     = w;
            output.dim[2].extent     = h;
        }

        if (0 == w || 0 == h) {
            return false;
        }
        outputs[0]->buffer().type = inputs[0]->getType();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto elementInM = (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
        auto interp     = op->main_as_Interp();
        auto unit       = 0;
        switch (interp->resizeType()) {
            case 1:
                unit = 1;
                break;
            case 2:
                unit = 4;
                break;
            case 3:
                unit = 16;
                break;
            default:
                break;
        }
        return unit * elementInM;
    }
};

REGISTER_SHAPE_INPUTS(InterpComputer, OpType_Interp, {1});
} // namespace MNN
