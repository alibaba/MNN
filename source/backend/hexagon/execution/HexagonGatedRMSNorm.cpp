//
//  HexagonGatedRMSNorm.cpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "HexagonGatedRMSNorm.hpp"
#include "HexagonBackend.hpp"
#include "backend/hexagon/backend/HexagonRuntime.hpp"
#include "core/OpCommonUtils.hpp"
#include "MNN_generated.h"
#include "htp_command.h"
#include "htp_gated_rms_norm_params.h"

namespace MNN {

HexagonGatedRMSNorm::HexagonGatedRMSNorm(std::shared_ptr<HexagonLayerNorm::Resource> res, Backend* backend)
    : HexagonExecution(backend), mResource(res) {}

bool HexagonGatedRMSNorm::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new HexagonGatedRMSNorm(mResource, bn);
    return true;
}

bool HexagonGatedRMSNorm::supported(const Op* op, const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs) {
#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && defined(MNN_GATED_RMS_NORM)
    int inside = 0;
    return OpCommonUtils::gatedRMSNormOpenCLOk(op, inputs, outputs, nullptr, &inside) &&
           inside % HTP_GATED_RMS_NORM_PACK == 0;
#else
    return false;
#endif
}

HexagonGatedRMSNorm* HexagonGatedRMSNorm::create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) {
    auto runtime = static_cast<const HexagonRuntime*>(backend->getRuntime());
    // The capability bit is the only thing that can still tell a skel without this kernel apart: the
    // geometry cannot ask the runtime, so it keeps the op whole on shape alone and refuses nothing here.
    if (runtime == nullptr || (runtime->info().capabilities & DSP_CAP_GATED_RMS_NORM) == 0 ||
        !supported(op, inputs, outputs)) {
        return nullptr;
    }
    auto res = HexagonLayerNorm::makeResource(backend, op);
    if (nullptr == res.get()) {
        return nullptr;
    }
    return new HexagonGatedRMSNorm(res, backend);
}

ErrorCode HexagonGatedRMSNorm::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                          std::vector<HexagonCommand>& dst) {
    // Shapes were validated by supported() at creation; x is [batch * heads, inside], z is [batch, heads * inside].
    DSPGatedRMSNormParam p;
    p.batch = inputs[1]->length(0);
    p.inside = inputs[0]->length(1);
    p.heads = inputs[1]->length(1) / p.inside;
    p.epsilon = mResource->mEpsilon;
    std::pair<int, int> gammaDev = {-1, 0};
    std::pair<int, int> betaDev = {-1, 0};
    if (mResource->mIniGammaBeta) {
        gammaDev = HexagonBackend::getDevicePtr(mResource->mGamma);
        if (!mResource->mBetaZero) {
            betaDev = HexagonBackend::getDevicePtr(mResource->mBeta);
        }
    }
    std::vector<std::pair<int, int>> inputFds = {HexagonBackend::getDevicePtr(inputs[0]),
                                                 HexagonBackend::getDevicePtr(inputs[1]), gammaDev, betaDev};
    std::vector<std::pair<int, int>> outputFds = {HexagonBackend::getDevicePtr(outputs[0])};
    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_GATED_RMS_NORM, &p, sizeof(p), inputFds, outputFds,
                     inputs, outputs);
    return NO_ERROR;
}

} // namespace MNN
