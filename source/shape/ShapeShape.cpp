//
//  ShapeShape.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <utility>
#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

static std::pair<int, int> _resolveShapeRange(const Op* op, int rank) {
    int start = 0;
    int end = rank;
    if (auto param = op->main_as_ShapeParam()) {
        if (param->hasStart()) {
            start = param->start();
            if (start < 0) {
                start += rank;
            }
        }
        if (param->hasEnd()) {
            end = param->end();
            if (end < 0) {
                end += rank;
            }
        }
    }
    start = std::max(0, std::min(start, rank));
    end = std::max(start, std::min(end, rank));
    return std::make_pair(start, end);
}

class ShapeSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 <= inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto& ib = inputs[0]->buffer();
        auto& ob = outputs[0]->buffer();

        ob.dimensions = 1;
        outputs[0]->setType(DataType_DT_INT32);
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = op->defaultDimentionFormat();
        auto inputFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        int rank = ib.dimensions;
        if (inputFormat == MNN_DATA_FORMAT_NC4HW4 && op->defaultDimentionFormat() == MNN_DATA_FORMAT_NHWC) {
            // For compability
            rank = 4;
        }
        auto range = _resolveShapeRange(op, rank);
        ob.dim[0].extent = range.second - range.first;
        return true;
    }
};

REGISTER_SHAPE(ShapeSizeComputer, OpType_Shape);

class ShapeRasterComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(1 == outputs.size());
        auto extra  = op->main_as_Extra();
        if (!extra) {
            // copy dims
            MNN_ASSERT(1 <= inputs.size());
            outputs[0]->buffer().type = inputs[0]->buffer().type;
            TensorUtils::copyShape(inputs[0], outputs[0], true);
        } else {
            if (inputs.size() > 0) {
                outputs[0]->buffer().type = inputs[0]->buffer().type;
                TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
            }
            for (int i = 0; i < extra->attr()->size(); i++) {
                auto attr = extra->attr()->Get(i);
                if (attr->key()->str() == "shape") {
                    outputs[0]->buffer().dimensions = 0;
                    if (attr->list()->i() != nullptr) {
                        int len = attr->list()->i()->size();
                        outputs[0]->buffer().dimensions = len;
                        for (int j = 0; j < len; j++) {
                            outputs[0]->setLength(j, attr->list()->i()->Get(j));
                        }
                    }
                    continue;
                }
                if (attr->key()->str() == "code") {
                    outputs[0]->buffer().type.code = (halide_type_code_t)attr->i();
                    continue;
                }
                if (attr->key()->str() == "bits") {
                    outputs[0]->buffer().type.bits = attr->i();
                    continue;
                }
                if (attr->key()->str() == "format") {
                    TensorUtils::getDescribe(outputs[0])->dimensionFormat = (MNN_DATA_FORMAT)attr->i();
                    continue;
                }
            }
        }
        return true;
    }
};

REGISTER_SHAPE(ShapeRasterComputer, OpType_Raster);
} // namespace MNN
