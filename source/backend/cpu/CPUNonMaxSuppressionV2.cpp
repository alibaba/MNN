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

#include "backend/cpu/CPUNonMaxSuppressionV2.hpp"
#include <math.h>
#include <queue>
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"

namespace MNN {

CPUNonMaxSuppressionV2::CPUNonMaxSuppressionV2(Backend* backend, const Op* op) : Execution(backend) {
    // nothing to do
}

// Return intersection-over-union overlap between boxes i and j
static inline float IOU(const float* boxes, int i, int j) {
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

void NonMaxSuppressionSingleClasssImpl(const Tensor* decodedBoxes, const float* scores, int maxDetections,
                                       float iouThreshold, float scoreThreshold, std::vector<int32_t>* selected) {
    MNN_ASSERT(iouThreshold >= 0.0f && iouThreshold <= 1.0f);
    MNN_ASSERT(decodedBoxes->dimensions() == 2);
    const int numBoxes = decodedBoxes->length(0);
    MNN_ASSERT(decodedBoxes->length(1) == 4)

    const int outputNum = std::min(maxDetections, numBoxes);
    std::vector<float> scoresData(numBoxes);
    std::copy_n(scores, numBoxes, scoresData.begin());

    struct Candidate {
        int boxIndex;
        float score;
    };

    auto cmp = [](const Candidate bsI, const Candidate bsJ) { return bsI.score < bsJ.score; };

    std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)> candidatePriorityQueue(cmp);

    for (int i = 0; i < scoresData.size(); ++i) {
        if (scoresData[i] > scoreThreshold) {
            candidatePriorityQueue.emplace(Candidate({i, scoresData[i]}));
        }
    }

    // std::vector<float> selectedScores;
    Candidate nextCandidate;
    float iou, originalScore;

    const auto boxesPtr = decodedBoxes->host<float>();
    while (selected->size() < outputNum && !candidatePriorityQueue.empty()) {
        nextCandidate = candidatePriorityQueue.top();
        originalScore = nextCandidate.score;
        candidatePriorityQueue.pop();

        // Overlapping boxes are likely to have similar scores,
        // therefore we iterate through the previously selected boxes backwards
        // in order to see if `next_candidate` should be suppressed.
        bool shouldSelect = true;
        for (int j = (int)selected->size() - 1; j >= 0; --j) {
            iou = IOU(boxesPtr, nextCandidate.boxIndex, selected->at(j));
            if (iou == 0.0) {
                continue;
            }
            if (iou > iouThreshold) {
                shouldSelect = false;
            }
        }

        if (shouldSelect) {
            selected->push_back(nextCandidate.boxIndex);
            // selectedScores.push_back(nextCandidate.score);
        }
    }
}

ErrorCode CPUNonMaxSuppressionV2::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    std::vector<int> selected;
    const int maxDetections    = inputs[2]->host<int32_t>()[0];
    float iouThreshold = 0, scoreThreshold = std::numeric_limits<float>::lowest();
    if (inputs.size() > 3) {
        iouThreshold   = inputs[3]->host<float>()[0];
    }
    if (inputs.size() > 4) {
        scoreThreshold = inputs[4]->host<float>()[0];
    }
    const auto scores          = inputs[1]->host<float>();
    NonMaxSuppressionSingleClasssImpl(inputs[0], scores, maxDetections, iouThreshold, scoreThreshold, &selected);
    std::copy_n(selected.begin(), selected.size(), outputs[0]->host<int32_t>());
    for (int i = selected.size(); i < outputs[0]->elementSize(); i++) {
        outputs[0]->host<int32_t>()[i] = -1;
    }

    return NO_ERROR;
}

class CPUNonMaxSuppressionV2Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUNonMaxSuppressionV2(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUNonMaxSuppressionV2Creator, OpType_NonMaxSuppressionV2);
} // namespace MNN
