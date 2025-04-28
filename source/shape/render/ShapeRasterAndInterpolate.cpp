//
//  ShapeRasterAndInterpolate.cpp
//  MNN
//
//  Created by MNN on 2023/02/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "math.h"

namespace MNN {
#ifdef MNN_SUPPORT_RENDER

class RasterAndInterpolateComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // Input: viewport ([x, y, w, h]), indice, position, attributes * n
        // Output: raster buffer: float batch, w, h, 1 attributes * n
        MNN_ASSERT(inputs.size() >= 2);
        int type = 4;
        if (op->main_type() == OpParameter_Extra) {
            auto extra = op->main_as_Extra();
            if (nullptr != extra->attr()) {
                for (int i=0; i<extra->attr()->size(); ++i) {
                    auto attr = extra->attr()->GetAs<Attribute>(i);
                    if (attr->key()->str() == "primitiveType") {
                        type = attr->i();
                        break;
                    }
                }
            }
        }
        auto format = TensorUtils::getDescribe(inputs[1])->dimensionFormat;
        if (type == 6) {
            auto numberPoint = inputs[0]->length(0);
            outputs[0]->buffer().dimensions = 0;
            outputs[0]->buffer().type = halide_type_of<int>();
            TensorUtils::getDescribe(outputs[0])->dimensionFormat = format;
            outputs[1]->buffer().dimensions = 2;
            outputs[1]->setLength(0, numberPoint);
            outputs[1]->setLength(1, 2);
            outputs[1]->buffer().type = halide_type_of<int>();
            TensorUtils::getDescribe(outputs[1])->dimensionFormat = format;
            return true;
        }
        if (type == 5) {
            auto pointSize = inputs[1];
            auto position = inputs[2];
            auto numberPoint = pointSize->length(0);
            auto color = inputs[3];
            auto conic = inputs[4];

            outputs[0]->buffer().dimensions = 0;
            outputs[0]->buffer().type = halide_type_of<int>();
            for (int i=1; i<outputs.size(); ++i) {
                outputs[i]->buffer().dimensions = 2;
                outputs[i]->setLength(0, numberPoint);
                outputs[i]->setLength(1, 4);
                outputs[i]->buffer().type = halide_type_of<float>();
                TensorUtils::getDescribe(outputs[i])->dimensionFormat = format;
            }
            return true;
        }
        auto indice = inputs[1];
        auto position = inputs[2];
        auto viewport = inputs[0];
        int width = viewport->host<int>()[2];
        int height = viewport->host<int>()[3];
        int batch = position->length(0);
        outputs[0]->buffer().dimensions    = 4;
        outputs[0]->setLength(0, batch);
        outputs[0]->setLength(1, height);
        outputs[0]->setLength(2, width);
        outputs[0]->setLength(3, 4); // traingle index, w0, w1, depth
        outputs[0]->buffer().type = halide_type_of<float>();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = format;
        for (int i=1; i<outputs.size(); ++i) {
            MNN_ASSERT(inputs[i+2]->dimensions() >= 2);
            int bpp = inputs[i+2]->length(inputs[i+2]->dimensions()-1);
            outputs[i]->buffer().dimensions = 4;
            outputs[i]->setLength(0, batch);
            outputs[i]->setLength(1, height);
            outputs[i]->setLength(2, width);
            outputs[i]->setLength(3, bpp);
            TensorUtils::getDescribe(outputs[i])->dimensionFormat = format;
        }
        return true;
    }
};

class TextureComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // Input: texture, uv, mipmap * n
        // Output: texels
        MNN_ASSERT(2 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto &ibInput0 = inputs[0]->buffer();
        auto &ob = outputs[0]->buffer();
        ob.type = ibInput0.type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(
                inputs[0])->dimensionFormat;
        if (op->main_as_GridSample()->backward()) {
            // For Grad, just copy the shape
            ob.dimensions = inputs[2]->length(0);
            auto shapePtr = inputs[2]->host<int>();
            for (int i=0; i<ob.dimensions; ++i) {
                ob.dim[i].extent = shapePtr[i];
            }
            return true;
        }
        int input_dim = inputs[0]->buffer().dimensions;
        int grid_dim = inputs[1]->buffer().dimensions;
        auto &ibInput1 = inputs[1]->buffer();

        ob.dimensions = ibInput1.dimensions;
        ob.dim[0].extent = ibInput0.dim[0].extent;
        ob.dim[3].extent = ibInput0.dim[ibInput0.dimensions - 1].extent;
        ob.dim[1].extent = ibInput1.dim[1].extent;
        ob.dim[2].extent = ibInput1.dim[2].extent;
        return true;
    }
};
#endif
REGISTER_SHAPE_INPUTS_RENDER(RasterAndInterpolateComputer, OpType_RasterAndInterpolate, (std::vector<int>{}));
REGISTER_SHAPE_INPUTS_RENDER(TextureComputer, OpType_Texture, (std::vector<int>{}));
} // namespace MNN
