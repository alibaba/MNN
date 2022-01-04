//
//  ShapeNonMaxSuppressionV2.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {

class NonMaxSuppressionV2Computer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        // boxes: [num_boxes, 4]
        const Tensor* boxes = inputs[0];
        // scores: [num_boxes]
        const Tensor* scores = inputs[1];
        // iou_threshold: scalar
        if (inputs.size() > 3 && inputs[3]->host<float>() != nullptr) {
            auto iou_threshold_val = inputs[3]->host<float>()[0];
            MNN_ASSERT(iou_threshold_val >= 0 && iou_threshold_val <= 1);
        }
        
        int num_boxes = 0;
        MNN_ASSERT(boxes->buffer().dimensions == 2);
        num_boxes = boxes->buffer().dim[0].extent;

        MNN_ASSERT(boxes->buffer().dimensions == 2 && scores->buffer().dim[0].extent == num_boxes &&
                   boxes->buffer().dim[1].extent == 4 && scores->buffer().dimensions == 1);

        int output_size = num_boxes;
        if (inputs.size() > 2 && inputs[2]->host<int32_t>() != nullptr) {
            output_size = std::min(inputs[2]->host<int32_t>()[0], num_boxes);
        }

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
