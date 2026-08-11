#include "HexagonRoPE.hpp"
#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "htp_command.h"

namespace MNN {

static bool _validRopeC4Input(const Tensor* q, const Tensor* k, int numHead, int kvNumHead, int headDim) {
    if (nullptr == q || nullptr == k || numHead <= 0 || kvNumHead <= 0 || headDim <= 0) {
        return false;
    }
    if (headDim % 64 != 0) {
        return false;
    }
    if (TensorUtils::getDescribe(q)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4 ||
        TensorUtils::getDescribe(k)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
        return false;
    }
    if (q->dimensions() < 2 || k->dimensions() < 2) {
        return false;
    }
    return q->length(1) == numHead * headDim && k->length(1) == kvNumHead * headDim;
}

static std::shared_ptr<HexagonLayerNorm::Resource> makeRopeNormResource(const LayerNorm* layerNormParam,
                                                                        Backend* backend) {
    if (layerNormParam == nullptr) {
        return nullptr;
    }
    std::shared_ptr<HexagonLayerNorm::Resource> res(new HexagonLayerNorm::Resource);
    res->mAllocator = static_cast<HexagonBackend*>(backend)->getAllocator(2);
    res->mAxis = layerNormParam->axis() != nullptr ? layerNormParam->axis()->size() : 0;
    res->mGroup = layerNormParam->group();
    res->mEpsilon = layerNormParam->epsilon();
    res->mRMSNorm = layerNormParam->useRMSNorm();
    auto gamma = layerNormParam->gamma();
    if (gamma == nullptr || gamma->size() <= 0) {
        return res;
    }
    auto pack = static_cast<const HexagonRuntime*>(backend->getRuntime())->info().vectorSize;
    const int gammaSize = gamma->size();
    const int allocSize = UP_DIV(gammaSize, pack) * pack * sizeof(float);
    res->mGamma = res->mAllocator->alloc(allocSize);
    res->mBeta = res->mAllocator->alloc(allocSize);
    if (res->mGamma.first == nullptr || res->mBeta.first == nullptr) {
        MNN_ERROR("Out of memory when gamma is acquired in Hexagon RoPE.\n");
        return nullptr;
    }
    auto gammaHost = reinterpret_cast<float*>(HexagonBackend::getPtr(res->mGamma));
    auto betaHost = reinterpret_cast<float*>(HexagonBackend::getPtr(res->mBeta));
    ::memset(gammaHost, 0, allocSize);
    ::memset(betaHost, 0, allocSize);
    ::memcpy(gammaHost, gamma->data(), gammaSize * sizeof(float));
    res->mIniGammaBeta = true;
    res->mBetaZero = true;
    static_cast<HexagonBackend*>(backend)->markHostInput(res->mGamma, allocSize);
    static_cast<HexagonBackend*>(backend)->markHostInput(res->mBeta, allocSize);
    return res;
}

HexagonRoPE::HexagonRoPE(Backend* bn, const Op* op) : HexagonExecution(bn) {
    auto param = op == nullptr ? nullptr : op->main_as_RoPEParam();
    if (param != nullptr) {
        mRopeCutHeadDim = param->rope_cut_head_dim();
        mNumHead = param->num_head();
        mKvNumHead = param->kv_num_head();
        mHeadDim = param->head_dim();
        mQNorm = makeRopeNormResource(param->q_norm(), backend());
        mKNorm = makeRopeNormResource(param->k_norm(), backend());
        mFuseLayerNorm = (mQNorm != nullptr || mKNorm != nullptr);
        return;
    }
    if (nullptr != op && OpParameter_Extra == op->main_type()) {
        auto extra = op->main_as_Extra();
        if (nullptr != extra && nullptr != extra->attr()) {
            for (int i = 0; i < extra->attr()->size(); ++i) {
                auto attr = extra->attr()->GetAs<Attribute>(i);
                if (nullptr == attr || nullptr == attr->key()) {
                    continue;
                }
                if (attr->key()->str() == "rope_cut_head_dim") {
                    mRopeCutHeadDim = attr->i();
                    continue;
                }
                if (attr->key()->str() == "num_head") {
                    mNumHead = attr->i();
                    continue;
                }
                if (attr->key()->str() == "kv_num_head") {
                    mKvNumHead = attr->i();
                    continue;
                }
                if (attr->key()->str() == "head_dim") {
                    mHeadDim = attr->i();
                    continue;
                }
                if (attr->key()->str() == "q_norm") {
                    auto qNormOp = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
                    mQNorm = HexagonLayerNorm::makeResource(backend(), qNormOp);
                    continue;
                }
                if (attr->key()->str() == "k_norm") {
                    auto kNormOp = flatbuffers::GetRoot<Op>(attr->tensor()->int8s()->data());
                    mKNorm = HexagonLayerNorm::makeResource(backend(), kNormOp);
                    continue;
                }
            }
        }
    }
    mFuseLayerNorm = (mQNorm != nullptr || mKNorm != nullptr);
}
HexagonRoPE::HexagonRoPE(Backend* bn):HexagonExecution(bn) {
    // Do nothing
}

HexagonRoPE* HexagonRoPE::create(Backend* backend, const Op* op) {
    if (op->type() != OpType_RoPE) {
        return nullptr;
    }
    auto functions = HexagonRuntime::getDstFunctions();
    if (functions == nullptr ) {
        return nullptr;
    }
    return new HexagonRoPE(backend, op);
}

ErrorCode HexagonRoPE::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  std::vector<HexagonCommand>& dst) {
    if (inputs.size() < 4 || outputs.size() < 2) {
        MNN_ERROR("[MNN::Hexagon] RoPE unsupported: inputs=%zu outputs=%zu\n", inputs.size(), outputs.size());
        return NOT_SUPPORT;
    }

    auto Q = inputs[0];
    auto K = inputs[1];
    auto cos = inputs[2];
    auto sin = inputs[3];

    auto Q_output = outputs[0];
    auto K_output = outputs[1];
    TensorUtils::getDescribe(Q_output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    TensorUtils::getDescribe(K_output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    for (auto t : inputs) {
        if (t != nullptr && HexagonBackend::getBytes(t) != 2) {
            MNN_ERROR("[MNN::Hexagon] RoPE unsupported: input bytes=%d dims=%d format=%d\n",
                      HexagonBackend::getBytes(t), t->dimensions(), TensorUtils::getDescribe(t)->dimensionFormat);
            return NOT_SUPPORT;
        }
    }
    for (auto t : outputs) {
        if (t != nullptr && HexagonBackend::getBytes(t) != 2) {
            MNN_ERROR("[MNN::Hexagon] RoPE unsupported: output bytes=%d dims=%d format=%d\n",
                      HexagonBackend::getBytes(t), t->dimensions(), TensorUtils::getDescribe(t)->dimensionFormat);
            return NOT_SUPPORT;
        }
    }

    if (!_validRopeC4Input(Q, K, mNumHead, mKvNumHead, mHeadDim)) {
        MNN_ERROR("[MNN::Hexagon] RoPE unsupported: invalid C4 q/k, qDim=%d kDim=%d qFmt=%d kFmt=%d qC=%d kC=%d heads=%d kvHeads=%d headDim=%d\n",
                  Q != nullptr ? Q->dimensions() : -1, K != nullptr ? K->dimensions() : -1,
                  Q != nullptr ? TensorUtils::getDescribe(Q)->dimensionFormat : -1,
                  K != nullptr ? TensorUtils::getDescribe(K)->dimensionFormat : -1,
                  Q != nullptr && Q->dimensions() > 1 ? Q->length(1) : -1,
                  K != nullptr && K->dimensions() > 1 ? K->length(1) : -1,
                  mNumHead, mKvNumHead, mHeadDim);
        return NOT_SUPPORT;
    }

    int batch = 1;
    int seqLen = Q->length(0);
    int numHead = mNumHead;
    int headDim = mHeadDim;
    int kvnumHead = mKvNumHead;

    int batch_seq = batch * seqLen;

    int ropeDim = mRopeCutHeadDim;
    if (ropeDim <= 0 || ropeDim > headDim) {
        ropeDim = headDim;
    }
    ropeDim = (ropeDim / 2) * 2;

    auto qIn = HexagonBackend::getDevicePtr(Q);
    auto kIn = HexagonBackend::getDevicePtr(K);
    auto hexagonBackend = static_cast<HexagonBackend*>(backend());
    auto cEven = HexagonBackend::getDevicePtr(cos);
    auto sEven = HexagonBackend::getDevicePtr(sin);

    auto qOut = HexagonBackend::getDevicePtr(Q_output);
    auto kOut = HexagonBackend::getDevicePtr(K_output);

    if (mFuseLayerNorm) {
        std::pair<int, int> qGammaDev = {-1, 0};
        std::pair<int, int> kGammaDev = {-1, 0};
        int qGammaSize = 0, kGammaSize = 0;

        if (mQNorm && mQNorm->mIniGammaBeta) {
            qGammaDev = HexagonBackend::getDevicePtr(mQNorm->mGamma);
            qGammaSize = headDim * sizeof(float);
        }
        if (mKNorm && mKNorm->mIniGammaBeta) {
            kGammaDev = HexagonBackend::getDevicePtr(mKNorm->mGamma);
            kGammaSize = headDim * sizeof(float);
        }

        struct RopeFuseParam {
            int batch_seq;
            int numHead;
            int kvnumHead;
            int headDim;
            int ropeDim;
            float qEps;
            float kEps;
            int qRmsNorm;
            int kRmsNorm;
            int inputC4;
        };

        float qEps = mQNorm ? mQNorm->mEpsilon : 1e-5f;
        float kEps = mKNorm ? mKNorm->mEpsilon : 1e-5f;
        int qRmsNorm = mQNorm ? (mQNorm->mRMSNorm ? 1 : 0) : 0;
        int kRmsNorm = mKNorm ? (mKNorm->mRMSNorm ? 1 : 0) : 0;

        RopeFuseParam params = {batch_seq, numHead, kvnumHead, headDim, ropeDim, qEps, kEps, qRmsNorm, kRmsNorm, 1};

        std::vector<std::pair<int, int>> inputFds = {qIn, kIn, cEven, sEven, qGammaDev, kGammaDev};
        std::vector<std::pair<int, int>> outputFds = {qOut, kOut};
        std::vector<Tensor*> commandInputs = {Q, K, cos, sin};
        commandInputs.emplace_back(nullptr);
        commandInputs.emplace_back(nullptr);

        dst.emplace_back();
        dst.back().build(hexagonBackend, DSP_OP_ROPE_FUSE_LAYERNORM, &params, sizeof(params),
                         inputFds,  outputFds,  commandInputs, outputs);
    } else {
        int params[] = {batch_seq, numHead, kvnumHead, headDim, ropeDim, 1};
        std::vector<std::pair<int, int>> inputFds = {qIn, kIn, cEven, sEven};
        std::vector<std::pair<int, int>> outputFds = {qOut, kOut};
        std::vector<Tensor*> commandInputs = {Q, K, cos, sin};

        dst.emplace_back();
        dst.back().build(hexagonBackend, DSP_OP_ROPE, params, sizeof(params),
                         inputFds,  outputFds,  commandInputs, outputs);
    }

    return NO_ERROR;
}

bool HexagonRoPE::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) return true;
    auto rope = new HexagonRoPE(bn);
    rope->mRopeCutHeadDim = mRopeCutHeadDim;
    rope->mNumHead = mNumHead;
    rope->mKvNumHead = mKvNumHead;
    rope->mHeadDim = mHeadDim;
    rope->mQNorm = mQNorm;
    rope->mKNorm = mKNorm;
    rope->mFuseLayerNorm = mFuseLayerNorm;
    *dst = rope;
    return true;
}

} // namespace MNN
