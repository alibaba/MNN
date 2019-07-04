//
//  CPUDetectionOutput.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUDetectionOutput.hpp"
#include <math.h>
#include <list>
#include "AutoTime.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "TensorUtils.hpp"

namespace MNN {

CPUDetectionOutput::CPUDetectionOutput(Backend *backend, int classCount, float nmsThreshold, int keepTopK,
                                       float confidenceThreshold, float objectnessScore)
    : Execution(backend),
      mClassCount(classCount),
      mNMSThreshold(nmsThreshold),
      mKeepTopK(keepTopK),
      mConfidenceThreshold(confidenceThreshold),
      mObjectnessScoreThreshold(objectnessScore) {
    // nothing to do
}

using score_box_t = std::tuple<float, float, float, float, int, float>;
#define box_rect(xmin, ymin, xmax, ymax, label, score) std::make_tuple((xmin), (ymin), (xmax), (ymax), (label), (score))
#define box_rect_xmin(rect) (std::get<0>(rect))
#define box_rect_ymin(rect) (std::get<1>(rect))
#define box_rect_xmax(rect) (std::get<2>(rect))
#define box_rect_ymax(rect) (std::get<3>(rect))
#define box_label(rect) (std::get<4>(rect))
#define box_score(rect) (std::get<5>(rect))

static inline float intersectionArea(score_box_t a, score_box_t b) {
    float axmin = box_rect_xmin(a), bxmin = box_rect_xmin(b);
    float axmax = box_rect_xmax(a), bxmax = box_rect_xmax(b);
    float aymin = box_rect_ymin(a), bymin = box_rect_ymin(b);
    float aymax = box_rect_ymax(a), bymax = box_rect_ymax(b);
    if (axmin > bxmax || axmax < bxmin || aymin > bymax || aymax < bymin)
        return 0.f;

    float interWidth  = fmin(axmax, bxmax) - fmax(axmin, bxmin);
    float interHeight = fmin(aymax, bymax) - fmax(aymin, bymin);
    return interWidth * interHeight;
}

static void pickBoxes(const std::vector<score_box_t> &boxes, std::list<long> &picked, float nmsThreshold) {
    long n = boxes.size();
    std::vector<float> areas;
    areas.resize(n);
    for (int i = 0; i < n; i++) {
        auto box     = boxes[i];
        float width  = box_rect_xmax(box) - box_rect_xmin(box);
        float height = box_rect_ymax(box) - box_rect_ymin(box);
        areas[i]     = width * height;
    }

    for (int i = 0; i < n; i++) {
        auto a = boxes[i];

        bool keep = true;
        for (auto pick : picked) {
            auto b = boxes[pick];

            // intersection over union
            float interArea = intersectionArea(a, b);
            float unionArea = areas[i] + areas[pick] - interArea;
            if (interArea / unionArea > nmsThreshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            picked.push_back(i);
        }
    }
}

ErrorCode CPUDetectionOutput::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // location transform space
    TensorUtils::copyShape(inputs[0], &mLocation);
    backend()->onAcquireBuffer(&mLocation, Backend::DYNAMIC);

    // confidence transform space
    TensorUtils::copyShape(inputs[1], &mConfidence);
    backend()->onAcquireBuffer(&mConfidence, Backend::DYNAMIC);

    // priorbox transform space
    TensorUtils::copyShape(inputs[2], &mPriorbox);
    backend()->onAcquireBuffer(&mPriorbox, Backend::DYNAMIC);

    // refine
    if (inputs.size() >= 5) {
        TensorUtils::copyShape(inputs[3], &mArmConfidence, false);
        TensorUtils::copyShape(inputs[4], &mArmLocation, false);

        backend()->onAcquireBuffer(&mArmConfidence, Backend::DYNAMIC);
        backend()->onAcquireBuffer(&mArmLocation, Backend::DYNAMIC);
        backend()->onReleaseBuffer(&mArmConfidence, Backend::DYNAMIC);
        backend()->onReleaseBuffer(&mArmLocation, Backend::DYNAMIC);
    }

    // release temp buffer space
    backend()->onReleaseBuffer(&mLocation, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mConfidence, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mPriorbox, Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUDetectionOutput::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &location   = inputs[0];
    auto &confidence = inputs[1];
    auto &priorbox   = inputs[2];
    auto &output     = outputs[0];

    // download
    MNNUnpackC4(mLocation.host<float>(), location->host<float>(), location->width() * location->height(),
                location->channel());
    MNNUnpackC4(mConfidence.host<float>(), confidence->host<float>(), confidence->width() * confidence->height(),
                confidence->channel());
    MNNUnpackC4(mPriorbox.host<float>(), priorbox->host<float>(), priorbox->width() * priorbox->height(),
                priorbox->channel());

    bool refineDet = inputs.size() >= 5;
    if (refineDet) {
        Tensor *armconfidence = inputs[3];
        Tensor *armlocation   = inputs[4];
        MNNUnpackC4(mArmConfidence.host<float>(), armconfidence->host<float>(),
                    armconfidence->width() * armconfidence->height(), armconfidence->channel());
        MNNUnpackC4(mArmLocation.host<float>(), armlocation->host<float>(),
                    armlocation->width() * armlocation->height(), armlocation->channel());
    }

    auto priorCount       = priorbox->height() / 4;
    auto locationPtr      = mLocation.host<const float>();
    auto confidencePtr    = mConfidence.host<const float>();
    auto priorboxPtr      = mPriorbox.host<const float>();
    auto variancePtr      = mPriorbox.host<const float>() + priorbox->height() * 1;
    auto armlocationPtr   = refineDet ? mArmLocation.host<const float>() : NULL;
    auto armconfidencePtr = refineDet ? mArmConfidence.host<const float>() : NULL;

    auto boxes      = std::shared_ptr<float>(new float[4 * priorCount], [](float *p) { delete[] p; });
    auto decodeBoxs = [&boxes, priorCount, variancePtr](const float *priorboxPtr, const float *locationPtr) {
        for (int i = 0; i < priorCount; i++) {
            auto loc = locationPtr + i * 4;
            auto pb  = priorboxPtr + i * 4;
            auto var = variancePtr + i * 4;
            auto box = boxes.get() + i * 4;

            float pbW  = pb[2] - pb[0];
            float pbH  = pb[3] - pb[1];
            float pbCX = (pb[0] + pb[2]) * 0.5f;
            float pbCY = (pb[1] + pb[3]) * 0.5f;

            float boxCX = var[0] * loc[0] * pbW + pbCX;
            float boxCY = var[1] * loc[1] * pbH + pbCY;
            float boxW  = exp(var[2] * loc[2]) * pbW;
            float boxH  = exp(var[3] * loc[3]) * pbH;

            box[0] = boxCX - boxW * 0.5f;
            box[1] = boxCY - boxH * 0.5f;
            box[2] = boxCX + boxW * 0.5f;
            box[3] = boxCY + boxH * 0.5f;
        }
    };
    if (refineDet) {
        decodeBoxs(priorboxPtr, armlocationPtr);
        decodeBoxs(boxes.get(), locationPtr);
    } else {
        decodeBoxs(priorboxPtr, locationPtr);
    }

    // sort and nms for each class
    std::vector<score_box_t> allClassBoxes;
    auto compareFunction = [](const score_box_t &a, const score_box_t &b) { return box_score(a) > box_score(b); };
    for (int i = 1; i < mClassCount; i++) { // start from 1 to ignore background class
        std::vector<score_box_t> classBoxes;

        // filter by confidenceThreshold
        for (int j = 0; j < priorCount; j++) {
            float score = confidencePtr[j * mClassCount + i];
            if (refineDet && (armconfidencePtr[j * 2 + 1] < mObjectnessScoreThreshold)) {
                score = 0.0;
            }
            if (score > mConfidenceThreshold) {
                const float *box = boxes.get() + 4 * j;
                classBoxes.push_back(box_rect(box[0], box[1], box[2], box[3], i, score));
            }
        }

        // sort inplace
        std::sort(classBoxes.begin(), classBoxes.end(), compareFunction);

        // apply nms
        std::list<long> picked;
        pickBoxes(classBoxes, picked, mNMSThreshold);

        // select
        for (auto index : picked) {
            allClassBoxes.push_back(classBoxes[index]);
        }
    }

    // global sort inplace
    std::sort(allClassBoxes.begin(), allClassBoxes.end(), compareFunction);

    // set width
    int numDetected = (int)allClassBoxes.size();
    if (numDetected > mKeepTopK) {
        numDetected = mKeepTopK;
    }
    output->buffer().dim[2].extent = numDetected;

    // write data
    auto outPtr = output->host<float>();
    for (int i = 0; i < numDetected; i++, outPtr += 6 * 4) {
        auto box      = allClassBoxes[i];
        outPtr[0 * 4] = box_label(box);
        outPtr[1 * 4] = box_score(box);
        outPtr[2 * 4] = box_rect_xmin(box);
        outPtr[3 * 4] = box_rect_ymin(box);
        outPtr[4 * 4] = box_rect_xmax(box);
        outPtr[5 * 4] = box_rect_ymax(box);
    }

    return NO_ERROR;
}

class CPUDetectionOutputCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto d = op->main_as_DetectionOutput();
        return new CPUDetectionOutput(backend, d->classCount(), d->nmsThresholdold(), d->keepTopK(),
                                      d->confidenceThreshold(), d->objectnessScore());
    }
};
REGISTER_CPU_OP_CREATOR(CPUDetectionOutputCreator, OpType_DetectionOutput);

} // namespace MNN
