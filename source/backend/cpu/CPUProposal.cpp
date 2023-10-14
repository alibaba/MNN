//
//  CPUProposal.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "backend/cpu/CPUProposal.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "CPUTensorConvert.hpp"
#include "core/TensorUtils.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {

CPUProposal::CPUProposal(Backend *backend, const Proposal *proposal) : Execution(backend), mProposal(proposal) {
    auto ratioCount = mProposal->ratios()->float32s()->size();
    auto numScale   = mProposal->scales()->float32s()->size();
    mAnchors.reset(4 * ratioCount * numScale);

    auto baseSize = mProposal->baseSize();
    const auto cx = baseSize * 0.5f;
    const auto cy = baseSize * 0.5f;
    auto ratios   = proposal->ratios()->float32s()->data();
    auto scales   = proposal->scales()->float32s()->data();
    auto anchors  = mAnchors.get();

    for (int i = 0; i < ratioCount; i++) {
        float ar = ratios[i];
        int rW   = round(baseSize / sqrt(ar));
        int rH   = round(rW * ar); // round(baseSize * sqrt(ar));

        for (int j = 0; j < numScale; j++) {
            float scale   = scales[j];
            float rsW     = rW * scale;
            float rsH     = rH * scale;
            float *anchor = anchors + 4 * (i * numScale + j);
            anchor[0]     = cx - rsW * 0.5f;
            anchor[1]     = cy - rsH * 0.5f;
            anchor[2]     = cx + rsW * 0.5f;
            anchor[3]     = cy + rsH * 0.5f;
        }
    }
}

using score_box_t = std::tuple<float, float, float, float, float>;
#define box_rect(xmin, ymin, xmax, ymax, score) std::make_tuple((xmin), (ymin), (xmax), (ymax), (score))
#define box_rect_xmin(box) (std::get<0>(box))
#define box_rect_ymin(box) (std::get<1>(box))
#define box_rect_xmax(box) (std::get<2>(box))
#define box_rect_ymax(box) (std::get<3>(box))
#define box_score(box) (std::get<4>(box))

static void pickBoxes(const std::vector<score_box_t> &boxes, std::vector<long> &picked, float nmsThreshold, int size) {
    long n = boxes.size();
    std::vector<float> areas;
    {
        areas.resize(n);
        for (int i = 0; i < n; i++) {
            auto box     = boxes[i];
            float width  = box_rect_xmax(box) - box_rect_xmin(box);
            float height = box_rect_ymax(box) - box_rect_ymin(box);
            areas[i]     = width * height;
        }
    }

    for (int i = 0; i < n; i++) {
        auto a = boxes[i];

        int keep = 1;
        for (int j = 0; j < picked.size(); j++) {
            auto b = boxes[picked[j]];

            // intersection over union
            float axmin = box_rect_xmin(a), bxmin = box_rect_xmin(b);
            float axmax = box_rect_xmax(a), bxmax = box_rect_xmax(b);
            float aymin = box_rect_ymin(a), bymin = box_rect_ymin(b);
            float aymax = box_rect_ymax(a), bymax = box_rect_ymax(b);
            if (axmin > bxmax || axmax < bxmin || aymin > bymax || aymax < bymin) {
                continue;
            }

            float interWidth  = fmin(axmax, bxmax) - fmax(axmin, bxmin);
            float interHeight = fmin(aymax, bymax) - fmax(aymin, bymin);
            float interArea   = interWidth * interHeight;
            float unionArea   = areas[i] + areas[picked[j]] - interArea;
            if (interArea / unionArea > nmsThreshold) {
                keep = 0;
                break;
            }
        }
        if (keep) {
            picked.emplace_back(i);
            if (picked.size() >= size) {
                break;
            }
        }
    }
}

ErrorCode CPUProposal::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto bufferAlloc = static_cast<CPUBackend *>(backend())->getBufferAllocator();
    mScoreBuffer = bufferAlloc->alloc(TensorUtils::getRawSize(inputs[0]) * inputs[0]->getType().bytes());
    if (mScoreBuffer.invalid()) {
        return OUT_OF_MEMORY;
    }
    // release temp buffer space
    bufferAlloc->free(mScoreBuffer);
    return NO_ERROR;
}

ErrorCode CPUProposal::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // score transform space
    auto score  = inputs[0];
    auto boxes  = inputs[1];
    auto imInfo = inputs[2];
    auto featStride   = mProposal->featStride();
    auto preNmsTopN   = mProposal->preNmsTopN();
    auto nmsThreshold = mProposal->nmsThreshold();
    auto afterNmsTopN = mProposal->afterNmsTopN();
    auto minSize      = mProposal->minSize();

    float* tmpScorePtr = (float*)mScoreBuffer.ptr();
    // download
    MNNUnpackC4Origin(tmpScorePtr, score->host<float>(), score->width() * score->height(), score->channel(), score->width() * score->height());

    auto scrWidth = score->width(), scrHeight = score->height(), scrSize = scrWidth * scrHeight;
    auto boxWidth = boxes->width(), boxHeight = boxes->height(), boxSize = boxWidth * boxHeight;
    auto imH = imInfo->host<float>()[0]; // NC/4HW4
    auto imW = imInfo->host<float>()[1]; // NC/4HW4

    // generate proposals from box deltas and shifted anchors
    // remove predicted boxes with either height or width < threshold
    auto anchorWidth  = 4;
    auto anchorHeight = mAnchors.size() / 4;
    std::vector<score_box_t> proposalBoxes;
    float imScale    = imInfo->host<float>()[2]; // NC/4HW4
    float minBoxSize = minSize * imScale;
    proposalBoxes.reserve(boxSize * anchorHeight);

    {
        for (int ah = 0; ah < anchorHeight; ++ah) {
            auto boxPtr   = boxes->host<float>() + ah * 4 * boxSize;
            auto scorePtr = tmpScorePtr + (ah + anchorHeight) * scrSize;

            // shifted anchor
            const auto anchor = mAnchors.get() + ah * anchorWidth;
            float anchorY     = anchor[1];
            float anchorW     = anchor[2] - anchor[0];
            float anchorH     = anchor[3] - anchor[1];

            for (int sh = 0; sh < scrHeight; sh++) {
                float anchorX = anchor[0];
                auto boxPtrH  = boxPtr + sh * 4 * boxWidth;

                for (int sw = 0; sw < scrWidth; sw++) {
                    auto box = boxPtrH + 4 * sw;
                    // apply center size
                    float cx = anchorX + anchorW * 0.5f + anchorW * box[0];
                    float cy = anchorY + anchorH * 0.5f + anchorH * box[1];
                    float w  = anchorW * exp(box[2]);
                    float h  = anchorH * exp(box[3]);

                    float minX = std::max(std::min(cx - w * 0.5f, imW - 1), 0.f);
                    float minY = std::max(std::min(cy - h * 0.5f, imH - 1), 0.f);
                    float maxX = std::max(std::min(cx + w * 0.5f, imW - 1), 0.f);
                    float maxY = std::max(std::min(cy + h * 0.5f, imH - 1), 0.f);
                    if (maxX - minX + 1 >= minBoxSize && maxY - minY + 1 >= minBoxSize) {
                        proposalBoxes.emplace_back(box_rect(minX, minY, maxX, maxY, scorePtr[sh * scrWidth + sw]));
                    }
                    anchorX += featStride;
                }
                anchorY += featStride;
            }
        }
    }

    {
        // sort all (proposal, score) pairs by score from highest to lowest
        // take top preNmsTopN
        auto compareFunction = [](const score_box_t &a, const score_box_t &b) {
            return box_score(a) > box_score(b);
        };
        if (0 < preNmsTopN && preNmsTopN < (int)proposalBoxes.size()) {
            std::partial_sort(proposalBoxes.begin(), proposalBoxes.begin() + preNmsTopN, proposalBoxes.end(),
                              compareFunction);
            proposalBoxes.resize(preNmsTopN);
        } else {
            std::sort(proposalBoxes.begin(), proposalBoxes.end(), compareFunction);
        }
    }

    // apply nms with nmsThreshold
    // take afterNmsTopN
    std::vector<long> picked;
    picked.reserve(afterNmsTopN);
    {
        pickBoxes(proposalBoxes, picked, nmsThreshold, afterNmsTopN);
    }

    int pickedCount = std::min((int)picked.size(), afterNmsTopN);

    // return the top proposals
    int roiStep = outputs[0]->buffer().dim[0].stride, scoreStep = 0;
    auto roiPtr = outputs[0]->host<float>(), scoresPtr = (float *)NULL;
    memset(roiPtr, 0, outputs[0]->size());

    if (outputs.size() > 1) {
        scoreStep = outputs[1]->buffer().dim[0].stride;
        scoresPtr = outputs[1]->host<float>();
        memset(scoresPtr, 0, outputs[1]->size());
    }

    for (int i = 0; i < pickedCount; i++, scoresPtr += scoreStep) {
        auto box  = proposalBoxes[picked[i]];
        roiPtr[i * 4 + 0] = 0;
        roiPtr[i * 4 + 1] = box_rect_xmin(box);
        roiPtr[i * 4 + 2] = box_rect_ymin(box);
        roiPtr[i * 4 + 3] = box_rect_xmax(box);
        roiPtr[i * 4 + outputs[0]->length(0) * 4] = box_rect_ymax(box);
        if (scoresPtr) {
            scoresPtr[0] = box_score(box);
        }
    }
    return NO_ERROR;
}

class CPUProposalCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        return new CPUProposal(backend, op->main_as_Proposal());
    }
};
REGISTER_CPU_OP_CREATOR(CPUProposalCreator, OpType_Proposal);

} // namespace MNN
