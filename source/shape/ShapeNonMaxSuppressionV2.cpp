//
//  ShapeNonMaxSuppressionV2.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Macro.h"
#include "core/SizeComputer.hpp"

namespace MNN {

class NonMaxSuppressionV2Computer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // boxes: [num_boxes, 4]
        const Tensor* boxes = inputs[0];
        // scores: [num_boxes]
        const Tensor* scores = inputs[1];
        // max_output_size: scalar
        const Tensor* max_output_size = inputs[2];
        // iou_threshold: scalar
        const Tensor* iou_threshold = inputs[3];

        const float iou_threshold_val = iou_threshold->host<float>()[0];

        MNN_ASSERT(iou_threshold_val >= 0 && iou_threshold_val <= 1);

        int num_boxes = 0;
        MNN_ASSERT(boxes->buffer().dimensions == 2);
        num_boxes = boxes->buffer().dim[0].extent;

        MNN_ASSERT(boxes->buffer().dimensions == 2 && scores->buffer().dim[0].extent == num_boxes &&
                   boxes->buffer().dim[1].extent == 4 && scores->buffer().dimensions == 1);

        const int output_size = std::min(max_output_size->host<int32_t>()[0], num_boxes);

        // TODO ramdom output shape only for fast rcnn
        outputs[0]->buffer().dimensions = 1;
        outputs[0]->setType(MNN::DataType_DT_INT32);
        outputs[0]->buffer().dim[0].extent = output_size;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE_INPUTS(NonMaxSuppressionV2Computer, OpType_NonMaxSuppressionV2, (std::vector<int>{2, 3}));
} // namespace MNN
