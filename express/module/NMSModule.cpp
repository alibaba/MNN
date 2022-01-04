//
//  NMSModule.cpp
//  MNN
//
//  Created by MNN on b'2020/09/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NMSModule.hpp"
#include "backend/cpu/CPUNonMaxSuppressionV2.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/Tensor.hpp>
#include "MNN_generated.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <queue>
namespace MNN {
namespace Express {

NMSModule* NMSModule::create(const Op* op) {
    auto module = new NMSModule;
    module->setType("NMSModule");
    if (nullptr != op->name()) {
        module->setName(op->name()->str());
    }
    return module;
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

static void NonMaxSuppressionSingleClasssImpl(const float* boxesPtr, const float* scores, int numBoxes, int maxDetections,
                                              float iouThreshold, float scoreThreshold, std::vector<int32_t>* selected) {
    MNN_ASSERT(iouThreshold >= 0.0f && iouThreshold <= 1.0f);

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

std::vector<Express::VARP> NMSModule::onForward(const std::vector<Express::VARP>& inputs) {
    const int maxDetections = inputs[2]->readMap<int>()[0];
    float iouThreshold = 0, scoreThreshold = std::numeric_limits<float>::lowest();
    if (inputs.size() > 3) {
        iouThreshold = inputs[3]->readMap<float>()[0];
    }
    if (inputs.size() > 4) {
        scoreThreshold = inputs[4]->readMap<float>()[0];
    }
    auto boxes = inputs[0], score = inputs[1];
    auto info = boxes->getInfo(), infoScore = score->getInfo();
    MNN_ASSERT(info->dim[info->dim.size() - 1] == 4);
    int batch = 1, numClass = 1, numBoxes = info->dim[0];
    bool onnxFormat = (infoScore->dim.size() > 1);
    if (onnxFormat) {
        batch = infoScore->dim[0];
        numClass = infoScore->dim[1];
        numBoxes = infoScore->dim[2];
    }
    INTS outputData;
    for (int b = 0; b < batch; ++b) {
        const auto boxesPtr = boxes->readMap<float>() + b * numBoxes * 4;
        for (int c = 0; c < numClass; ++c) {
            std::vector<int> selected;
            const auto scorePtr = score->readMap<float>() + (b * numClass + c) * numBoxes;
            NonMaxSuppressionSingleClasssImpl(boxesPtr, scorePtr, numBoxes, maxDetections, iouThreshold, scoreThreshold, &selected);
            for (int i = 0; i < selected.size(); ++i) {
                if (onnxFormat) {
                    outputData.push_back(b);
                    outputData.push_back(c);
                }
                outputData.push_back(selected[i]);
            }
        }
    }
    
    Variable::Info outInfo;
    outInfo.order = info->order;
    outInfo.type = halide_type_of<int>();
    if (onnxFormat) {
        outInfo.dim.assign({(int)outputData.size() / 3, 3});
    } else {
        outInfo.dim.assign({(int)outputData.size()});
    }
    outInfo.syncSize();
    VARPS outputs;
    outputs.push_back(Variable::create(Expr::create(std::move(outInfo), outputData.data(), VARP::CONSTANT)));
    return outputs;
}

Module* NMSModule::clone(CloneContext* ctx) const {
    NMSModule* module(new NMSModule);
    return this->cloneBaseTo(ctx, module);
}

}
}
