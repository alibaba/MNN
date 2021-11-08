//
//  ShapeDetectionPostProcess.cpp
//  MNN
//
//  Created by MNN on 2019/10/29.
//  Copyright © 2018, Alibaba Group Holding Limited

#include "shape/SizeComputer.hpp"

namespace MNN {

class DetectionPostProcessSize : public SizeComputer {
public:
    bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs) const override {
        MNN_CHECK(inputs.size() == 3, "DetectionPostProcess should have 3 inputs!");
        MNN_CHECK(outputs.size() == 4, "DetectionPostProcess should have 4 outputs!");

        auto postProcess           = op->main_as_DetectionPostProcessParam();
        const int numDetectedBoxes = postProcess->maxDetections() * postProcess->maxClassesPerDetection();

        const int bathSize = inputs[0]->batch();

        // Outputs: detection_boxes, detection_scores, detection_classes,
        // num_detections
        auto detectionBoxes                 = outputs[0];
        detectionBoxes->buffer().dimensions = 3;
        detectionBoxes->setLength(0, bathSize);
        detectionBoxes->setLength(1, numDetectedBoxes);
        detectionBoxes->setLength(2, 4);
        detectionBoxes->buffer().type = halide_type_of<float>();

        auto detectionClasses                 = outputs[1];
        detectionClasses->buffer().dimensions = 2;
        detectionClasses->setLength(0, bathSize);
        detectionClasses->setLength(1, numDetectedBoxes);
        detectionClasses->buffer().type = halide_type_of<float>();

        auto detectionScores                 = outputs[2];
        detectionScores->buffer().dimensions = 2;
        detectionScores->setLength(0, bathSize);
        detectionScores->setLength(1, numDetectedBoxes);
        detectionScores->buffer().type = halide_type_of<float>();

        auto numDetections                 = outputs[3];
        numDetections->buffer().dimensions = 1;
        numDetections->setLength(0, 1);
        numDetections->buffer().type = halide_type_of<float>();

        return true;
    }
};

REGISTER_SHAPE(DetectionPostProcessSize, OpType_DetectionPostProcess);

} // namespace MNN
