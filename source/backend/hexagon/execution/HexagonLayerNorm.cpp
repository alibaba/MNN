//
//  HexagonLayerNorm.cpp
//  MNN
//
//  Created by MNN on 2025/04/28
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "HexagonLayerNorm.hpp"
#include "HexagonBackend.hpp"
#include "backend/hexagon/backend/HexagonRuntime.hpp"
#include "backend/hexagon/execution/HexagonRaster.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "htp_command.h"

namespace MNN {

HexagonLayerNorm::HexagonLayerNorm(std::shared_ptr<Resource> res, Backend* backend) : HexagonExecution(backend), mResource(res) {
    mAllocator = static_cast<HexagonBackend*>(backend)->getAllocator(1);
}

HexagonLayerNorm::~HexagonLayerNorm() {
}

bool HexagonLayerNorm::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new HexagonLayerNorm(mResource, bn);
    return true;
}

std::shared_ptr<HexagonLayerNorm::Resource> HexagonLayerNorm::makeResource(Backend* backend, const MNN::Op* op) {
    const auto* layer_norm_param = op->main_as_LayerNorm();
    std::shared_ptr<HexagonLayerNorm::Resource> res(new Resource);
    res->mAllocator = static_cast<HexagonBackend*>(backend)->getAllocator(2);
    res->mAxis = 0;
    if (nullptr != layer_norm_param->axis()) {
        res->mAxis = layer_norm_param->axis()->size();
    }
    auto pack = static_cast<const HexagonRuntime*>(backend->getRuntime())->info().vectorSize;
    res->mGroup = layer_norm_param->group();
    res->mEpsilon = layer_norm_param->epsilon();
    res->mRMSNorm = layer_norm_param->useRMSNorm();
    bool hasGammaBeta = layer_norm_param->gamma() != nullptr;
    int gammasize = 0;
    if (hasGammaBeta) {
        gammasize = layer_norm_param->gamma()->size();
    }
    hasGammaBeta = hasGammaBeta || (layer_norm_param->external() && layer_norm_param->external()->size() > 1 && layer_norm_param->external()->data()[1] > 0);
    if (hasGammaBeta && gammasize == 0) {
        gammasize = layer_norm_param->external()->data()[1] / sizeof(float);
    }
    if (hasGammaBeta) {
        res->mIniGammaBeta = true;
        res->mGamma = res->mAllocator->alloc(UP_DIV(gammasize, pack) * pack * sizeof(float));
        res->mBeta = res->mAllocator->alloc(UP_DIV(gammasize, pack) * pack * sizeof(float));
        if (res->mGamma.first == nullptr || res->mBeta.first == nullptr) {
            MNN_ERROR("Out of memory when gamma is acquired in HexagonLayerNorm.\n");
            return nullptr;
        }

        float* gamma_data_host = (float*)HexagonBackend::getPtr(res->mGamma);
        float* beta_data_host = (float*)HexagonBackend::getPtr(res->mBeta);
        ::memset(gamma_data_host, 0, UP_DIV(gammasize, pack) * pack * sizeof(float));
        ::memset(beta_data_host, 0, UP_DIV(gammasize, pack) * pack * sizeof(float));

        if (layer_norm_param->gamma()) {
            memcpy(gamma_data_host, layer_norm_param->gamma()->data(), gammasize * sizeof(float));
        }
        if (layer_norm_param->beta()) {
            memcpy(beta_data_host, layer_norm_param->beta()->data(), gammasize * sizeof(float));
        }
        res->mBetaZero = true;
        for (int i = 0; i < gammasize; ++i) {
            if (beta_data_host[i] != 0.0f) {
                res->mBetaZero = false;
                break;
            }
        }
        auto hexagonBackend = static_cast<HexagonBackend*>(backend);
        hexagonBackend->markHostInput(res->mGamma, UP_DIV(gammasize, pack) * pack * (int)sizeof(float));
        hexagonBackend->markHostInput(res->mBeta, UP_DIV(gammasize, pack) * pack * (int)sizeof(float));
    }
    return res;
}

HexagonLayerNorm* HexagonLayerNorm::create(Backend* backend, const MNN::Op* op) {
    auto res = makeResource(backend, op);
    if (nullptr == res.get()) {
        return nullptr;
    }
    return new HexagonLayerNorm(res, backend);
}

ErrorCode HexagonLayerNorm::onBuildCmd(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs,
                                       std::vector<HexagonCommand>& dst) {
    mOutterSize = 1;
    mInnerSize = 1;
    const auto layout = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

    do {
        int rank = inputs.at(0)->dimensions();
        if (mResource->mGroup > 1) {
            mOutterSize = inputs.at(0)->length(0) * mResource->mGroup;
            for (int i = 1; i < rank; i++) {
                mInnerSize *= inputs.at(0)->length(i);
            }
            mInnerSize /= mResource->mGroup;
            break;
        }
        for (int i = 0; i < rank - mResource->mAxis; ++i) {
            mOutterSize *= inputs.at(0)->length(i);
        }
        for (int i = rank - mResource->mAxis; i < rank; ++i) {
            mInnerSize *= inputs.at(0)->length(i);
        }
    } while (false);

    auto input = inputs[0];
    auto output = outputs[0];
    auto srcDev = HexagonBackend::getDevicePtr(input);
    auto dstDev = HexagonBackend::getDevicePtr(output);

    std::pair<int, int> gammaDev = {-1, 0};
    std::pair<int, int> betaDev = {-1, 0};
    if (mResource->mIniGammaBeta) {
        gammaDev = HexagonBackend::getDevicePtr(mResource->mGamma);
        if (!mResource->mBetaZero) {
            betaDev = HexagonBackend::getDevicePtr(mResource->mBeta);
        }
    }

    if (layout == MNN_DATA_FORMAT_NC4HW4 && inputs.size() == 2 && outputs.size() == 2) {
        auto input1 = inputs[1];
        auto output1 = outputs[1];
        auto src1Dev = HexagonBackend::getDevicePtr(input1);
        auto dst1Dev = HexagonBackend::getDevicePtr(output1);

        struct AddFuseLayerNormParam {
            int batch;
            int channels;
            float epsilon;
            int rmsNorm;
        };

        int area = input->batch();
        for (int i=2; i<input->dimensions(); ++i) {
            area *= input->length(i);
        }
        int channels = input->channel();
        AddFuseLayerNormParam params = {area, channels, mResource->mEpsilon, mResource->mRMSNorm ? 1 : 0};

        std::vector<std::pair<int, int>> inputFds = {srcDev, src1Dev, gammaDev, betaDev};
        std::vector<std::pair<int, int>> outputFds = {dst1Dev, dstDev};

        dst.emplace_back();
        dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_ADD_FUSE_LAYERNORM, &params, sizeof(params),
                         inputFds,  outputFds,  inputs, outputs);
        return NO_ERROR;
    }

    struct LayerNormParam {
        int dim0;
        int dim1;
        float epsilon;
        int rmsNorm;
    };

    int dim0 = layout == MNN_DATA_FORMAT_NC4HW4 ? input->batch() : mOutterSize;
    int dim1 = layout == MNN_DATA_FORMAT_NC4HW4 ? input->channel() : mInnerSize;
    LayerNormParam params = {dim0, dim1, mResource->mEpsilon, mResource->mRMSNorm ? 1 : 0};

    std::vector<std::pair<int, int>> inputFds = {srcDev, gammaDev, betaDev};
    std::vector<std::pair<int, int>> outputFds = {dstDev};

    int opType = layout == MNN_DATA_FORMAT_NC4HW4 ? DSP_OP_LAYER_NORM_PACKED : DSP_OP_LAYER_NORM;
    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), opType, &params, sizeof(params),
                     inputFds,  outputFds,  inputs, outputs);

    return NO_ERROR;
}

} // namespace MNN
