/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/

// edited from tensorflow - non_max_suppression_op.cc by MNN.

#include "CPUNonMaxSuppressionV2.hpp"
#include <math.h>
#include <queue>
#include "CPUBackend.hpp"
#include "Macro.h"

namespace MNN {

template <typename T>
CPUNonMaxSuppressionV2<T>::CPUNonMaxSuppressionV2(Backend* backend, const Op* op) : Execution(backend) {
    // nothing to do
}

// Return intersection-over-union overlap between boxes i and j
static inline float IOU(float* boxes, int i, int j) {
    const float yMinI = std::min<float>(boxes[i * 4 + 0], boxes[i * 4 + 2]);
    const float xMinI = std::min<float>(boxes[i * 4 + 1], boxes[i * 4 + 3]);
    const float yMaxI = std::max<float>(boxes[i * 4 + 0], boxes[i * 4 + 2]);
    const float xMaxI = std::max<float>(boxes[i * 4 + 1], boxes[i * 4 + 3]);
    const float yMinJ = std::min<float>(boxes[j * 4 + 0], boxes[j * 4 + 2]);
    const float xMinJ = std::min<float>(boxes[j * 4 + 1], boxes[j * 4 + 3]);
    const float yMaxJ = std::max<float>(boxes[j * 4 + 0], boxes[j * 4 + 2]);
    const float xMaxJ = std::max<float>(boxes[j * 4 + 1], boxes[j * 4 + 3]);
    const float areaI = (yMaxI - yMinI) * (xMaxI - xMinI);
    const float areaJ = (yMaxJ - yMinJ) * (xMaxJ - xMinJ);
    if (areaI <= 0 || areaJ <= 0)
        return 0.0;
    const float intersectionYMin = std::max<float>(yMinI, yMinJ);
    const float intersectionXMin = std::max<float>(xMinI, xMinJ);
    const float intersectionYMax = std::min<float>(yMaxI, yMaxJ);
    const float intersectionXMax = std::min<float>(xMaxI, xMaxJ);
    const float intersectionArea = std::max<float>(intersectionYMax - intersectionYMin, 0.0) *
                                   std::max<float>(intersectionXMax - intersectionXMin, 0.0);
    return intersectionArea / (areaI + areaJ - intersectionArea);
}

template <typename T>
ErrorCode CPUNonMaxSuppressionV2<T>::onExecute(const std::vector<Tensor*>& inputs,
                                               const std::vector<Tensor*>& outputs) {
    // boxes: [num_boxes, 4]
    const Tensor* boxes = inputs[0];
    // scores: [num_boxes]
    const Tensor* scores = inputs[1];
    // max_output_size: scalar
    const Tensor* maxOutputSize = inputs[2];
    // iou_threshold: scalar
    const Tensor* iouThreshold = inputs[3];

    const float iouThresholdVal   = iouThreshold->host<float>()[0];
    const float scoreThresholdVal = std::numeric_limits<float>::lowest();

    MNN_ASSERT(iouThresholdVal >= 0 && iouThresholdVal <= 1);
    MNN_ASSERT(boxes->buffer().dimensions == 2);
    int numBoxes = boxes->buffer().dim[0].extent;

    MNN_ASSERT(boxes->buffer().dimensions == 2 && scores->buffer().dim[0].extent == numBoxes &&
               boxes->buffer().dim[1].extent == 4 && scores->buffer().dimensions == 1);

    const int outputSize = std::min(maxOutputSize->host<int32_t>()[0], numBoxes);

    std::vector<float> scoresData(numBoxes);

    std::copy_n(scores->host<float>(), numBoxes, scoresData.begin());

    struct Candidate {
        int boxIndex;
        float score;
    };

    auto cmp = [](const Candidate bsI, const Candidate bsJ) { return bsI.score < bsJ.score; };

    std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)> candidatePriorityQueue(cmp);

    for (int i = 0; i < scoresData.size(); ++i) {
        if (scoresData[i] > scoreThresholdVal) {
            candidatePriorityQueue.emplace(Candidate({i, scoresData[i]}));
        }
    }

    std::vector<int> selected;
    std::vector<float> selectedScores;
    Candidate nextCandidate;
    float iou, originalScore;

    while (selected.size() < outputSize && !candidatePriorityQueue.empty()) {
        nextCandidate = candidatePriorityQueue.top();
        originalScore = nextCandidate.score;
        candidatePriorityQueue.pop();

        // Overlapping boxes are likely to have similar scores,
        // therefore we iterate through the previously selected boxes backwards
        // in order to see if `next_candidate` should be suppressed.
        bool shouldSelect = true;
        for (int j = (int)selected.size() - 1; j >= 0; --j) {
            iou = IOU(boxes->host<float>(), nextCandidate.boxIndex, selected[j]);
            if (iou == 0.0) {
                continue;
            }
            if (iou > iouThresholdVal) {
                shouldSelect = false;
            }
        }

        if (shouldSelect) {
            selected.push_back(nextCandidate.boxIndex);
            selectedScores.push_back(nextCandidate.score);
        }
    }

    std::copy_n(selected.begin(), selected.size(), outputs[0]->host<int32_t>());

    return NO_ERROR;
}

class CPUNonMaxSuppressionV2Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUNonMaxSuppressionV2<int32_t>(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUNonMaxSuppressionV2Creator, OpType_NonMaxSuppressionV2);
} // namespace MNN
