//
//  CPUROIPooling.cpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUROIPooling.hpp"
#include <float.h>
#include <math.h>
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"
#include "TensorUtils.hpp"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

CPUROIPooling::CPUROIPooling(Backend *backend, int pooledWidth, int pooledHeight, float spatialScale)
    : Execution(backend), mPooledWidth(pooledWidth), mPooledHeight(pooledHeight), mSpatialScale(spatialScale) {
    // nothing to do
}

ErrorCode CPUROIPooling::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // roi transform space
    auto &roi = inputs[1]->buffer();
    memcpy(mROI.buffer().dim, roi.dim, sizeof(halide_dimension_t) * roi.dimensions);
    mROI.buffer().dim[1].flags = 0;
    TensorUtils::setLinearLayout(&mROI);
    backend()->onAcquireBuffer(&mROI, Backend::DYNAMIC);

    // release temp buffer space
    backend()->onReleaseBuffer(&mROI, Backend::DYNAMIC);

    return NO_ERROR;
}

static inline int max(int a, int b) {
    return a > b ? a : b;
}
static inline int min(int a, int b) {
    return a < b ? a : b;
}

ErrorCode CPUROIPooling::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &input  = inputs[0];
    auto &output = outputs[0];

    // download
    for (int i = 0; i < mROI.batch(); ++i) {
        auto &roi = inputs[1];
        MNNUnpackC4(mROI.host<float>() + i * mROI.buffer().dim[0].stride,
                    roi->host<float>() + i * roi->buffer().dim[0].stride, roi->width() * roi->height(), roi->channel());
    }

    // get params
    auto iw = input->width(), ih = input->height(), is = iw * ih * 4;
    auto ow = output->width(), oh = output->height(), os = ow * oh * 4;
    auto slice     = UP_DIV(input->channel(), 4);
    auto numROI    = inputs[1]->batch();
    auto batchSize = input->batch();

    for (int n = 0; n < numROI; ++n) {
        auto batchOutput = output->host<float>() + output->buffer().dim[0].stride * n;
        auto roiPtr      = mROI.host<float>() + mROI.buffer().dim[0].stride * n;
        int roi          = roiPtr[0];
        int x1           = round(roiPtr[1] * mSpatialScale);
        int y1           = round(roiPtr[2] * mSpatialScale);
        int x2           = round(roiPtr[3] * mSpatialScale);
        int y2           = round(roiPtr[4] * mSpatialScale);
        MNN_ASSERT(roi < batchSize);

        int roiW = max(x2 - x1 + 1, 1);
        int roiH = max(y2 - y1 + 1, 1);

        float binSizeW = (float)roiW / (float)mPooledWidth;
        float binSizeH = (float)roiH / (float)mPooledHeight;

        auto batchInput = input->host<float>() + input->buffer().dim[0].stride * roi;
        for (int s = 0; s < slice; s++) {
            auto sliceInput = batchInput + is * s;
            auto rowOutput  = batchOutput + os * s;
            float binPh     = 0;
            for (int ph = 0; ph < mPooledHeight; ph++, rowOutput += mPooledWidth * 4) {
                // Compute pooling region for this output unit:
                //  start (included) = floor(ph * roiHeight / pooledHeight)
                //  end (excluded) = ceil((ph + 1) * roiHeight / pooledHeight)
                int hStart = min(max(y1 + (int)floorf(binPh), 0), ih);
                binPh += binSizeH;
                int hEnd = min(max(y1 + (int)ceilf(binPh), 0), ih);
                int hLen = hEnd - hStart;
                if (hLen <= 0) {
                    memset(rowOutput, 0, mPooledWidth * 4 * sizeof(float));
                    continue;
                }

                float binPw = 0;
                for (int pw = 0; pw < mPooledWidth; pw++) {
                    int wStart = min(max(x1 + (int)floorf(binPw), 0), iw);
                    binPw += binSizeW;
                    int wEnd = min(max(x1 + (int)ceilf(binPw), 0), iw);
                    int wLen = wEnd - wStart;
                    if (wLen <= 0) {
                        memset(rowOutput + pw * 4, 0, 4 * sizeof(float));
                        continue;
                    }

#ifdef MNN_USE_NEON
                    auto ptr        = sliceInput + (hStart * iw + wStart) * 4;
                    float32x4_t max = vdupq_n_f32(-FLT_MAX);
                    // float32x4_t max = vdupq_n_f32(-MAXFLOAT);
                    for (int h = 0; h < hLen; h++, ptr += iw * 4) {
                        for (int w = 0; w < wLen; w++) {
                            float32x4_t in = vld1q_f32(ptr + w * 4);
                            max            = vmaxq_f32(max, in);
                        }
                    }
                    vst1q_f32(rowOutput + pw * 4, max);
#else
                    for (int i = 0; i < 4; i++) {
                        auto ptr  = sliceInput + (hStart * iw + wStart) * 4 + i;
                        float max = -FLT_MAX;
                        for (int h = 0; h < hLen; h++, ptr += iw * 4) {
                            for (int w = 0; w < wLen; w++) {
                                max = std::max(max, ptr[w * 4]);
                            }
                        }
                        rowOutput[pw * 4 + i] = max;
                    }
#endif
                }
            }
        }
    }

    return NO_ERROR;
}

class CPUROIPoolingCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto roi = op->main_as_RoiPooling();
        return new CPUROIPooling(backend, roi->pooledWidth(), roi->pooledHeight(), roi->spatialScale());
    }
};
REGISTER_CPU_OP_CREATOR(CPUROIPoolingCreator, OpType_ROIPooling);

} // namespace MNN
