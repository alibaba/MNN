//
//  ShapeInterp.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

// Size Computer
class Interp3DComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == inputs.size() || 2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());

        auto& input         = inputs[0]->buffer(); // input tensor(data)
        auto& output        = outputs[0]->buffer();
        int w               = 0;
        int h               = 0;
        int d               = 0;
        const int inputSize = (int)inputs.size();
        auto shape = inputs[0]->shape();
        auto iw = shape[4];
        auto ih = shape[3];
        auto id = shape[2];
        // copy dims
        memcpy(output.dim, input.dim, sizeof(halide_dimension_t) * input.dimensions);
        outputs[0]->buffer().dimensions = inputs[0]->dimensions();
        outputs[0]->buffer().type = inputs[0]->getType();
        auto format = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = format;
        if (2 == inputSize) {
            auto shape = inputs[1]; // input shape(shape)
            if(shape->length(0) == input.dimensions) {
                // For Onnx's Resize
                // Don't support batch / channel resize
                for (int i=0; i<2; ++i) {
                    output.dim[i].extent = input.dim[i].extent;
                }
                if (shape->getType().code == halide_type_int) {
                    // Width / Height
                    auto shapePtr = shape->host<int>();
                    for (int i=2; i<input.dimensions; ++i) {
                        output.dim[i].extent = shapePtr[i];
                    }
                } else {
                    // Scale
                    auto scalePtr = shape->host<float>();
                    for (int i=2; i<input.dimensions; ++i) {
                        output.dim[i].extent = (scalePtr[i] * (float)input.dim[i].extent);
                    }
                }
                return true;
            }
        }
        if (1 == inputSize) {
            auto interp3D = op->main_as_Interp3D();
            // get output dims
            w = interp3D->outputWidth();
            h = interp3D->outputHeight();
            d = interp3D->outputDepth();
            if (w == 0 || h == 0) {
                w = iw * interp3D->widthScale();
                h = ih * interp3D->heightScale();
                d = id * interp3D->depthScale();
            }
        } else {
            auto shape = inputs[1]; // input shape(shape)
            // Tensorflow's interp: d, h, w
            if(3 != shape->buffer().dim[0].extent) {
                MNN_ERROR("Tensorflow's interp's shape should be length two\n");
                return false;
            }
            if (shape->getType().code == halide_type_float) {
                const float *shapeData = shape->host<float>();
                w                      = shapeData[2];
                h                      = shapeData[1];
                d                      = shapeData[0];
            } else {
                const int32_t *shapeData = shape->host<int32_t>();
                w                        = shapeData[2];
                h                        = shapeData[1];
                d                        = shapeData[0];
            }
        }
        if (0 == w && 0 == h) {
            return false;
        }
        MNN_ASSERT(MNN_DATA_FORMAT_NHWC != format)
        if (MNN_DATA_FORMAT_NHWC == format) {
            //output.dim[2].extent     = w;
            //output.dim[1].extent     = h;
            MNN_ERROR("NHWC input format is not supported in Interp3D currently.");
        } else {
            output.dim[4].extent     = w;
            output.dim[3].extent     = h;
            output.dim[2].extent     = d;
        }
        return true;
    }
    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto elementInM = (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
        auto interp3D     = op->main_as_Interp3D();
        auto unit       = 0;
        switch (interp3D->resizeType()) {
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

REGISTER_SHAPE_INPUTS(Interp3DComputer, OpType_Interp3D, {1});
} // namespace MNN
