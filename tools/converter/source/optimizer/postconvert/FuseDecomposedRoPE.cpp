//
//  FuseDecomposedRoPE.cpp
//  MNNConverter
//
//  Created by MNN on 2026/11/02.
//

#include <cmath>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <MNN/MNNDefine.h>
#include "../PostTreatUtils.hpp"
#include "GraphRewrite.hpp"

using namespace MNN;

namespace {

static bool readConstScalar(const GraphView& g, int tensor, int32_t& value) {
    auto op = g.producerOp(tensor);
    if (op == nullptr || op->type != OpType_Const || op->main.type != OpParameter_Blob) {
        return false;
    }
    auto blob = op->main.AsBlob();
    if (blob == nullptr || blob->int32s.size() != 1) {
        return false;
    }
    value = blob->int32s[0];
    return true;
}

static MNN::OpT* asBinary(MNN::OpT* op, BinaryOpOperation type) {
    if (op == nullptr || op->type != OpType_BinaryOp || op->main.type != OpParameter_BinaryOp) {
        return nullptr;
    }
    auto binary = op->main.AsBinaryOp();
    if (binary == nullptr || binary->opType != type || binary->activationType != 0 ||
        op->inputIndexes.size() != 2) {
        return nullptr;
    }
    return op;
}

static bool ropeTableUsable(const GraphView& g, int tensor, int headDim) {
    auto op = g.producerOp(tensor);
    if (op == nullptr || op->type != OpType_Const || op->main.type != OpParameter_Blob) {
        return true;
    }
    auto blob = op->main.AsBlob();
    if (blob == nullptr || blob->dims.empty()) {
        return true;
    }
    int64_t rows = 1;
    for (int i = 0; i + 1 < (int)blob->dims.size(); ++i) {
        rows *= blob->dims[i];
    }
    return rows > 1 && blob->dims.back() == headDim;
}

static bool matchSliceParams(const GraphView& g, int sliceTensor, int xTensor, int xRank, int32_t& begin, int32_t& end,
                             int32_t& steps) {
    auto slice = g.producerOp(sliceTensor);
    if (slice == nullptr || slice->type != OpType_StridedSlice || slice->inputIndexes.size() != 5 ||
        slice->main.type != OpParameter_StridedSliceParam) {
        return false;
    }
    auto param = slice->main.AsStridedSliceParam();
    if (param == nullptr || param->fromType != 1 || slice->inputIndexes[0] != xTensor) {
        return false;
    }
    int32_t axes;
    if (!readConstScalar(g, slice->inputIndexes[1], begin) || !readConstScalar(g, slice->inputIndexes[2], end) ||
        !readConstScalar(g, slice->inputIndexes[3], axes) || !readConstScalar(g, slice->inputIndexes[4], steps)) {
        return false;
    }
    if (axes != -1 && axes != xRank - 1) {
        return false;
    }
    return true;
}

static bool sliceCoversFullHead(int32_t end, int headDim) {
    return end >= headDim || end >= (1 << 29);
}

static bool matchInterleavedRotate(const GraphView& g, const MNN::OpT* rotConcat, int xTensor, int xRank, int headDim) {
    auto axis = rotConcat->main.AsAxis();
    if (rotConcat->inputIndexes.size() != 2 || (axis->axis != -1 && axis->axis != xRank)) {
        return false;
    }
    int oddSliceOut = -1, evenSliceOut = -1;
    for (int k = 0; k < 2; ++k) {
        auto unsqueeze = g.producerOp(rotConcat->inputIndexes[k]);
        if (unsqueeze == nullptr || unsqueeze->type != OpType_Unsqueeze || unsqueeze->inputIndexes.size() < 2) {
            return false;
        }
        int32_t unsqueezeAxis;
        if (!readConstScalar(g, unsqueeze->inputIndexes[1], unsqueezeAxis) ||
            (unsqueezeAxis != -1 && unsqueezeAxis != xRank)) {
            return false;
        }
        if (k == 0) {
            auto neg = g.producerOp(unsqueeze->inputIndexes[0]);
            if (neg == nullptr || neg->type != OpType_UnaryOp || neg->main.type != OpParameter_UnaryOp ||
                neg->main.AsUnaryOp()->opType != UnaryOpOperation_NEG || neg->inputIndexes.size() != 1) {
                return false;
            }
            oddSliceOut = neg->inputIndexes[0];
        } else {
            evenSliceOut = unsqueeze->inputIndexes[0];
        }
    }
    int32_t begin, end, steps;
    if (!matchSliceParams(g, oddSliceOut, xTensor, xRank, begin, end, steps) || begin != 1 || steps != 2 ||
        !sliceCoversFullHead(end, headDim)) {
        return false;
    }
    if (!matchSliceParams(g, evenSliceOut, xTensor, xRank, begin, end, steps) || begin != 0 || steps != 2 ||
        !sliceCoversFullHead(end, headDim)) {
        return false;
    }
    return true;
}

static bool matchHalfSplitRotate(const GraphView& g, const MNN::OpT* rotConcat, int xTensor, int xRank, int headDim) {
    auto axis = rotConcat->main.AsAxis();
    if (rotConcat->inputIndexes.size() != 2 || (axis->axis != -1 && axis->axis != xRank - 1)) {
        return false;
    }
    if (headDim % 2 != 0) {
        return false;
    }
    auto neg = g.producerOp(rotConcat->inputIndexes[0]);
    if (neg == nullptr || neg->type != OpType_UnaryOp || neg->main.type != OpParameter_UnaryOp ||
        neg->main.AsUnaryOp()->opType != UnaryOpOperation_NEG || neg->inputIndexes.size() != 1) {
        return false;
    }
    const int half = headDim / 2;
    int32_t begin, end, steps;
    if (!matchSliceParams(g, neg->inputIndexes[0], xTensor, xRank, begin, end, steps) || begin != half || steps != 1 ||
        !sliceCoversFullHead(end, headDim)) {
        return false;
    }
    if (!matchSliceParams(g, rotConcat->inputIndexes[1], xTensor, xRank, begin, end, steps) || begin != 0 ||
        end != half || steps != 1) {
        return false;
    }
    return true;
}

struct RotationMatch {
    bool valid = false;
    bool interleaved = false;
    int addOpIndex = -1;
    int addTensor = -1;
    int xTensor = -1;
    int cosTensor = -1;
    int sinTensor = -1;
    int numHead = 0;
    int headDim = 0;
    int xRank = 0;
    int preNormIndex = -1;
    int geometryTensor = -1;
};

static RotationMatch matchRotation(const GraphView& g, int addOpIndex) {
    RotationMatch match;
    auto add = asBinary(g.opAt(addOpIndex), BinaryOpOperation_ADD);
    if (add == nullptr || add->outputIndexes.size() != 1) {
        return match;
    }
    auto mul0 = asBinary(g.producerOp(add->inputIndexes[0]), BinaryOpOperation_MUL);
    auto mul1 = asBinary(g.producerOp(add->inputIndexes[1]), BinaryOpOperation_MUL);
    if (mul0 == nullptr || mul1 == nullptr) {
        return match;
    }
    int xTensor = -1, cosTensor = -1;
    MNN::OpT* mulCos = nullptr;
    MNN::OpT* mulSin = nullptr;
    for (auto mul : {mul0, mul1}) {
        for (int k = 0; k < 2; ++k) {
            int t = mul->inputIndexes[k];
            if (t < 0 || t >= (int)g.consumers.size()) {
                continue;
            }
            int sliceConsumers = 0;
            for (int c : g.consumers[t]) {
                auto cop = g.opAt(c);
                if (cop != nullptr && cop->type == OpType_StridedSlice) {
                    ++sliceConsumers;
                }
            }
            if (sliceConsumers >= 2) {
                if (xTensor >= 0) {
                    return match;
                }
                xTensor = t;
                cosTensor = mul->inputIndexes[1 - k];
                mulCos = mul;
                mulSin = (mul == mul0) ? mul1 : mul0;
            }
        }
    }
    if (xTensor < 0) {
        return match;
    }
    int xSliceConsumers = 0;
    for (int c : g.consumers[xTensor]) {
        auto cop = g.opAt(c);
        if (cop == nullptr) {
            return match;
        }
        if (cop == mulCos || cop->type == OpType_Shape || cop->type == OpType_Rank || cop->type == OpType_Size) {
            continue;
        }
        if (cop->type == OpType_StridedSlice) {
            ++xSliceConsumers;
            continue;
        }
        return match;
    }
    if (xSliceConsumers != 2) {
        return match;
    }
    int sinTensor = -1;
    const MNN::OpT* rotConcat = nullptr;
    bool interleaved = false;
    for (int k = 0; k < 2 && rotConcat == nullptr; ++k) {
        int t = mulSin->inputIndexes[k];
        auto prodOp = g.producerOp(t);
        if (prodOp == nullptr) {
            continue;
        }
        const MNN::OpT* cc = prodOp;
        bool viaReshape = false;
        if (prodOp->type == OpType_Reshape && !prodOp->inputIndexes.empty()) {
            viaReshape = true;
            cc = g.producerOp(prodOp->inputIndexes[0]);
        }
        if (cc != nullptr && cc->type == OpType_Concat && cc->main.type == OpParameter_Axis) {
            rotConcat = cc;
            interleaved = viaReshape;
            sinTensor = mulSin->inputIndexes[1 - k];
        }
    }
    if (rotConcat == nullptr || sinTensor < 0 || sinTensor == cosTensor) {
        return match;
    }
    int geometryTensor = xTensor;
    int preNormIndex = -1;
    auto xProducer = g.producerOp(xTensor);
    if (xProducer != nullptr && xProducer->type == OpType_LayerNorm && xProducer->main.type == OpParameter_LayerNorm &&
        xProducer->inputIndexes.size() == 1) {
        preNormIndex = g.producerOf(xTensor);
        geometryTensor = xProducer->inputIndexes[0];
    }
    auto xReshape = g.producerOp(geometryTensor);
    if (xReshape == nullptr || xReshape->type != OpType_Reshape || xReshape->inputIndexes.size() != 2) {
        return match;
    }
    auto shapeConcat = g.producerOp(xReshape->inputIndexes[1]);
    if (shapeConcat == nullptr || shapeConcat->type != OpType_Concat || shapeConcat->main.type != OpParameter_Axis ||
        shapeConcat->main.AsAxis()->axis != 0) {
        return match;
    }
    const int arity = (int)shapeConcat->inputIndexes.size();
    int32_t batch = 1, numHead, headDim;
    bool batchConst = readConstScalar(g, shapeConcat->inputIndexes[0], batch);
    if (arity != 4 || (batchConst && batch != 1) ||
        !readConstScalar(g, shapeConcat->inputIndexes[2], numHead) ||
        !readConstScalar(g, shapeConcat->inputIndexes[3], headDim)) {
        return match;
    }
    if (numHead <= 0 || headDim <= 0 || (headDim % 2) != 0) {
        return match;
    }
    const int xRank = arity;
    bool rotateOk = interleaved ? matchInterleavedRotate(g, rotConcat, xTensor, xRank, headDim)
                                : matchHalfSplitRotate(g, rotConcat, xTensor, xRank, headDim);
    if (!rotateOk) {
        return match;
    }
    if (!ropeTableUsable(g, cosTensor, headDim) || !ropeTableUsable(g, sinTensor, headDim)) {
        return match;
    }
    match.valid = true;
    match.interleaved = interleaved;
    match.addOpIndex = addOpIndex;
    match.addTensor = add->outputIndexes[0];
    match.xTensor = xTensor;
    match.cosTensor = cosTensor;
    match.sinTensor = sinTensor;
    match.numHead = numHead;
    match.headDim = headDim;
    match.xRank = xRank;
    match.preNormIndex = preNormIndex;
    match.geometryTensor = geometryTensor;
    return match;
}

static bool normFoldable(const MNN::OpT* op, int xRank, int headDim) {
    if (op == nullptr || op->type != OpType_LayerNorm || op->main.type != OpParameter_LayerNorm ||
        op->inputIndexes.size() != 1 || op->outputIndexes.size() != 1) {
        return false;
    }
    auto norm = op->main.AsLayerNorm();
    if (norm == nullptr || !norm->useRMSNorm || norm->group != 1 || !norm->beta.empty() || !norm->external.empty() ||
        norm->axis.size() != 1 || (norm->axis[0] != -1 && norm->axis[0] != xRank - 1)) {
        return false;
    }
    if (!norm->gamma.empty() && (int)norm->gamma.size() != headDim) {
        return false;
    }
    return true;
}

static std::unique_ptr<LayerNormT> cloneNorm(const MNN::OpT* op) {
    auto src = op->main.AsLayerNorm();
    std::unique_ptr<LayerNormT> n(new LayerNormT);
    n->axis = src->axis;
    n->epsilon = src->epsilon;
    n->gamma = src->gamma;
    n->group = src->group;
    n->useRMSNorm = src->useRMSNorm;
    return n;
}

static bool rotationPreservesNorm(const GraphView& g, int cosTensor, int sinTensor, int headDim, bool interleaved) {
    auto readTable = [&](int tensor, const std::vector<float>*& data, int64_t& rows) -> bool {
        auto op = g.producerOp(tensor);
        if (op == nullptr || op->type != OpType_Const || op->main.type != OpParameter_Blob) {
            return false;
        }
        auto blob = op->main.AsBlob();
        if (blob == nullptr || blob->dims.empty() || blob->dims.back() != headDim) {
            return false;
        }
        rows = 1;
        for (int i = 0; i + 1 < (int)blob->dims.size(); ++i) {
            rows *= blob->dims[i];
        }
        if (rows <= 0 || blob->float32s.size() != (size_t)(rows * headDim)) {
            return false;
        }
        data = &blob->float32s;
        return true;
    };
    const std::vector<float>* cosData = nullptr;
    const std::vector<float>* sinData = nullptr;
    int64_t cosRows = 0, sinRows = 0;
    if (!readTable(cosTensor, cosData, cosRows) || !readTable(sinTensor, sinData, sinRows) || cosRows != sinRows) {
        return false;
    }
    const float tol = 1e-4f;
    const int half = headDim / 2;
    for (int64_t r = 0; r < cosRows; ++r) {
        for (int p = 0; p < half; ++p) {
            int i0 = interleaved ? (int)(r * headDim + 2 * p) : (int)(r * headDim + p);
            int i1 = interleaved ? i0 + 1 : i0 + half;
            float c0 = (*cosData)[i0], s0 = (*sinData)[i0];
            float c1 = (*cosData)[i1], s1 = (*sinData)[i1];
            if (std::fabs(c0 * c0 + s0 * s0 - 1.0f) > tol || std::fabs(c1 * c1 + s1 * s1 - 1.0f) > tol) {
                return false;
            }
            const bool same = std::fabs(c1 - c0) <= tol && std::fabs(s1 - s0) <= tol;
            const bool negated = std::fabs(c1 + c0) <= tol && std::fabs(s1 + s0) <= tol;
            if (!same && !negated) {
                return false;
            }
        }
    }
    return true;
}

static int findC4Source(const GraphView& g, int tensor, int expectChannels) {
    std::set<int> visited;
    bool c4Handoff = false;
    while (true) {
        int p = g.producerOf(tensor);
        if (p < 0 || !visited.insert(p).second) {
            return -1;
        }
        auto op = g.opAt(p);
        if (op->type == OpType_Convolution && op->main.type == OpParameter_Convolution2D) {
            auto conv = op->main.AsConvolution2D();
            if (c4Handoff && conv != nullptr && conv->common != nullptr &&
                conv->common->outputCount == expectChannels && op->outputIndexes.size() == 1) {
                return op->outputIndexes[0];
            }
            return -1;
        }
        if (op->type == OpType_ConvertTensor && op->main.type == OpParameter_TensorConvertInfo) {
            auto info = op->main.AsTensorConvertInfo();
            if (info == nullptr || info->source != MNN_DATA_FORMAT_NC4HW4) {
                return -1;
            }
            c4Handoff = true;
        } else if (op->type != OpType_Reshape) {
            return -1;
        }
        if (op->inputIndexes.empty()) {
            return -1;
        }
        tensor = op->inputIndexes[0];
    }
}

enum class NormOrder { Either, Before, After };

struct SideFusion {
    int outputTensor = -1;
    int sourceTensor = -1;
    bool foldNorm = false;
    NormOrder normOrder = NormOrder::Either;
    int postNormOpIndex = -1;
    std::unique_ptr<LayerNormT> normParam;
};

static SideFusion decideSide(const GraphView& g, const RotationMatch& rot, int postNormIndex) {
    SideFusion side;
    side.sourceTensor = findC4Source(g, rot.geometryTensor, rot.numHead * rot.headDim);
    if (side.sourceTensor < 0) {
        return side;
    }
    if (rot.preNormIndex >= 0) {
        auto normOp = g.opAt(rot.preNormIndex);
        if (!normFoldable(normOp, rot.xRank, rot.headDim)) {
            return side;
        }
        side.foldNorm = true;
        side.normOrder = NormOrder::Before;
        side.normParam = cloneNorm(normOp);
        side.outputTensor = rot.addTensor;
        return side;
    }
    if (postNormIndex >= 0) {
        auto normOp = g.opAt(postNormIndex);
        const auto& users = g.consumers[rot.addTensor];
        bool soleConsumer = users.size() == 1 && users[0] == postNormIndex;
        if (soleConsumer && normFoldable(normOp, rot.xRank, rot.headDim)) {
            auto norm = normOp->main.AsLayerNorm();
            bool commutable = norm->gamma.empty() &&
                              rotationPreservesNorm(g, rot.cosTensor, rot.sinTensor, rot.headDim, rot.interleaved);
            side.foldNorm = true;
            side.normOrder = commutable ? NormOrder::Either : NormOrder::After;
            side.postNormOpIndex = postNormIndex;
            side.normParam = cloneNorm(normOp);
            side.outputTensor = normOp->outputIndexes[0];
            return side;
        }
    }
    side.outputTensor = rot.addTensor;
    return side;
}

}

class FuseDecomposedRoPE : public PostConverter {
public:
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        int fused = 0;
        while (true) {
            GraphView g(net.get());
            bool progressed = false;
            for (int i = 0; i < (int)net->oplists.size() && !progressed; ++i) {
                auto attention = net->oplists[i].get();
                if (attention->type != OpType_Attention || attention->inputIndexes.size() < 2) {
                    continue;
                }
                int qAnchor = g.producerOf(attention->inputIndexes[0]);
                int kAnchor = g.producerOf(attention->inputIndexes[1]);
                if (qAnchor < 0 || kAnchor < 0 || qAnchor == kAnchor) {
                    continue;
                }
                int qPostNorm = -1, kPostNorm = -1;
                int qAdd = qAnchor, kAdd = kAnchor;
                auto qAnchorOp = g.opAt(qAnchor);
                auto kAnchorOp = g.opAt(kAnchor);
                if (qAnchorOp->type == OpType_LayerNorm && qAnchorOp->inputIndexes.size() == 1) {
                    qPostNorm = qAnchor;
                    qAdd = g.producerOf(qAnchorOp->inputIndexes[0]);
                }
                if (kAnchorOp->type == OpType_LayerNorm && kAnchorOp->inputIndexes.size() == 1) {
                    kPostNorm = kAnchor;
                    kAdd = g.producerOf(kAnchorOp->inputIndexes[0]);
                }
                auto qRot = matchRotation(g, qAdd);
                auto kRot = matchRotation(g, kAdd);
                if (!qRot.valid || !kRot.valid) {
                    continue;
                }
                if (qRot.cosTensor != kRot.cosTensor || qRot.sinTensor != kRot.sinTensor ||
                    qRot.headDim != kRot.headDim || qRot.interleaved != kRot.interleaved) {
                    continue;
                }
                // Folding pre-norm relies on the dead-op sweep, which is skipped for nets with subgraphs.
                if (!net->subgraphs.empty() && (qRot.preNormIndex >= 0 || kRot.preNormIndex >= 0)) {
                    continue;
                }
                auto qSide = decideSide(g, qRot, qPostNorm);
                auto kSide = decideSide(g, kRot, kPostNorm);
                if (qSide.outputTensor < 0 || kSide.outputTensor < 0) {
                    continue;
                }
                const bool needsPreNorm = qSide.normOrder == NormOrder::Before || kSide.normOrder == NormOrder::Before;
                const bool needsPostNorm = qSide.normOrder == NormOrder::After || kSide.normOrder == NormOrder::After;
                if (needsPreNorm && needsPostNorm) {
                    continue;
                }
                int lastProducer = std::max({g.producerOf(qSide.sourceTensor), g.producerOf(kSide.sourceTensor),
                                             g.producerOf(qRot.cosTensor), g.producerOf(qRot.sinTensor)});
                OpReplacement rep;
                rep.insertPos = lastProducer + 1;
                rep.removedOps = {qRot.addOpIndex, kRot.addOpIndex};
                // Pre-norm still feeds the now-dead rotation chain; leave it to sweepUnreachable.
                if (qSide.postNormOpIndex >= 0) {
                    rep.removedOps.push_back(qSide.postNormOpIndex);
                }
                if (kSide.postNormOpIndex >= 0) {
                    rep.removedOps.push_back(kSide.postNormOpIndex);
                }
                rep.op.reset(new MNN::OpT);
                rep.op->type = OpType_RoPE;
                rep.op->name = attention->name + "/FusedRoPE";
                rep.op->inputIndexes = {qSide.sourceTensor, kSide.sourceTensor, qRot.cosTensor, qRot.sinTensor};
                rep.op->outputIndexes = {qSide.outputTensor, kSide.outputTensor};
                rep.op->main.type = OpParameter_RoPEParam;
                rep.op->main.value = new RoPEParamT;
                auto param = rep.op->main.AsRoPEParam();
                param->rope_cut_head_dim = qRot.headDim;
                param->num_head = qRot.numHead;
                param->kv_num_head = kRot.numHead;
                param->head_dim = qRot.headDim;
                param->interleaved = qRot.interleaved;
                param->norm_after = needsPostNorm;
                if (qSide.foldNorm) {
                    param->q_norm = std::move(qSide.normParam);
                }
                if (kSide.foldNorm) {
                    param->k_norm = std::move(kSide.normParam);
                }
                rep.op->defaultDimentionFormat = MNN_DATA_FORMAT_NHWC;
                std::vector<OpReplacement> reps;
                reps.push_back(std::move(rep));
                if (applyReplacements(net.get(), g, reps)) {
                    ++fused;
                    progressed = true;
                }
            }
            if (!progressed) {
                break;
            }
        }
        if (fused == 0) {
            return true;
        }
        int removed = 0;
        if (net->subgraphs.empty()) {
            removed = sweepUnreachable(net.get());
        }
        MNN_PRINT("FuseDecomposedRoPE: fused %d attention(s), removed %d dead ops\n", fused, removed);
        return true;
    }
};

static PostConverterRegister<FuseDecomposedRoPE> __l("FuseDecomposedRoPE");