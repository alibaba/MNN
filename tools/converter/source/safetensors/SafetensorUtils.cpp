#include "SafetensorUtils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include "MNN_generated.h"
#include "core/IDSTEncoder.hpp"

namespace MNN {
namespace Express {
namespace SafeTensorUtils {

VARP _MakeLastHiddenStateOutput(VARP hiddenState, int hiddenSize) {
    if (nullptr == hiddenState.get()) {
        return nullptr;
    }
    std::vector<int> sizes = {1, 1, hiddenSize};
    auto sizeVar = _Const(sizes.data(), {3}, NCHW, halide_type_of<int>());
    std::vector<int> begins = {0, -1, 0};
    auto beginVar = _Const(begins.data(), {3}, NCHW, halide_type_of<int>());
    auto output = _Slice(hiddenState, beginVar, sizeVar);
    output->setName("last_hidden_state");
    return output;
}

VARP _GPT2Attention(int numHead, int headDim, VARP q, VARP k, VARP v, VARP qk_scale_q, VARP qk_scale_k,
                    VARP sv_scale_s, VARP sv_scale_v, VARP mask, bool supportC4Opt, float attnScale) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_Attention;
    op->main.value = new AttentionParamT;
    op->main.type = OpParameter_AttentionParam;
    op->main.AsAttentionParam()->kv_cache = true;
    op->main.AsAttentionParam()->attnScale = attnScale;
    bool supportC4 = (headDim % 16 == 0) && supportC4Opt;
    op->main.AsAttentionParam()->output_c4 = supportC4;
    if (nullptr != qk_scale_q || nullptr != qk_scale_k) {
        op->main.AsAttentionParam()->mhq_quant.resize(4);
        for (int i = 0; i < 4; ++i) {
            op->main.AsAttentionParam()->mhq_quant[i].reset(new TensorQuantInfoT);
            op->main.AsAttentionParam()->mhq_quant[i]->scale = 0.0f;
        }
        auto& mhqQuant = op->main.AsAttentionParam()->mhq_quant;
        if (nullptr != qk_scale_q) {
            mhqQuant[0]->scale = qk_scale_q->readMap<float>()[0];
        }
        if (nullptr != qk_scale_k) {
            mhqQuant[1]->scale = qk_scale_k->readMap<float>()[0];
        }
        if (nullptr != sv_scale_s) {
            mhqQuant[2]->scale = sv_scale_s->readMap<float>()[0];
        }
        if (nullptr != sv_scale_v) {
            mhqQuant[3]->scale = sv_scale_v->readMap<float>()[0];
        }
    }
    VARP output;
    if (nullptr != mask.get()) {
        output = Variable::create(Expr::create(op.get(), {q, k, v, mask}));
    } else {
        output = Variable::create(Expr::create(op.get(), {q, k, v}));
    }
    if (!supportC4) {
        output = _Reshape(output, {-1, numHead * headDim, 1, 1});
    }
    return output;
}

static void _splitBufToArray(const uint8_t* buf, uint8_t* arr, size_t arrLen, size_t needBits) {
    unsigned char mask = (1 << needBits) - 1;
    unsigned char* tmp = (unsigned char*)buf;
    int offset = 0;
    for (size_t i = 0; i < arrLen; ++i) {
        unsigned char idx = 0;
        long shift = 8 - needBits - offset % 8;
        if (shift < 0) {
            idx = (tmp[offset / 8] << (0 - shift)) & mask;
            idx |= (tmp[(offset / 8) + 1] >> (8 + shift)) & mask;
        } else {
            idx = (tmp[offset / 8] >> shift) & mask;
        }
        offset += needBits;
        if (offset % 8 == 0) {
            tmp += offset / 8;
            offset = 0;
        }
        arr[i] = idx;
    }
}

VARP _QConvolution1x1(int inputCount, VARP input, VARP inputScale, VARP inputZero, VARP weight, VARP wscale,
                      VARP wzeropoint, VARP bias, int outputCount, bool scaleInputCount, int weightBit) {
    std::unique_ptr<OpT> conv(new OpT);
    conv->type = OpType_Convolution;
    conv->main.type = OpParameter_Convolution2D;
    conv->main.value = new Convolution2DT;
    auto parm = conv->main.AsConvolution2D();
    parm->common.reset(new Convolution2DCommonT);
    if (outputCount > 0) {
        parm->common->outputCount = outputCount;
    } else {
        parm->common->outputCount = (int)bias->getInfo()->size;
        outputCount = parm->common->outputCount;
    }
    auto weightSize = weight->getInfo()->size;
    auto weightInputCount = weightSize / parm->common->outputCount;
    if (0 == weightBit) {
        weightBit = 8 * (int)weightInputCount / inputCount;
    }
    MNN_ASSERT(weightBit <= 8);
    if (nullptr == wscale.get()) {
        parm->weight.resize(weightSize);
        auto ptr = weight->readMap<float>();
        if (nullptr == ptr) {
            MNN_ERROR("_QConvolution1x1: weight->readMap<float>() is nullptr!\n");
            return nullptr;
        }
        ::memcpy(parm->weight.data(), ptr, weightSize * sizeof(float));
        parm->common->inputCount = inputCount;
        parm->bias.resize(parm->common->outputCount);
        if (nullptr != bias) {
            auto bptr = bias->readMap<float>();
            if (nullptr == bptr) {
                MNN_ERROR("_QConvolution1x1: bias->readMap<float>() is nullptr!\n");
                return nullptr;
            }
            ::memcpy(parm->bias.data(), bptr, bias->getInfo()->size * sizeof(float));
        } else {
            ::memset(parm->bias.data(), 0, parm->bias.size() * sizeof(float));
        }
        return Variable::create(Expr::create(conv.get(), {input}));
    }

    if (scaleInputCount) {
        std::vector<float> scales(inputCount);
        auto scalePtr = wscale->readMap<float>();
        if (nullptr == scalePtr) {
            MNN_ERROR("_QConvolution1x1: wscale->readMap<float>() is nullptr!\n");
            return nullptr;
        }
        ::memcpy(scales.data(), scalePtr, inputCount * sizeof(float));
        std::vector<float> emptyBias;
        input = _Scale(input, inputCount, std::move(scales), std::move(emptyBias));
        wscale = _Const(1.0f, {outputCount}, NCHW);
    }
    if (wscale->getInfo()->size == 1 && parm->common->outputCount > 1) {
        auto scalePtr = wscale->readMap<float>();
        if (nullptr == scalePtr) {
            MNN_ERROR("_QConvolution1x1: scalar wscale->readMap<float>() is nullptr!\n");
            return nullptr;
        }
        std::vector<float> scales(parm->common->outputCount, scalePtr[0]);
        wscale = _Const(scales.data(), {parm->common->outputCount}, NCHW);
    }
    auto scaleSize = wscale->getInfo()->size;
    if (parm->common->outputCount > scaleSize) {
        MNN_ERROR("scaleSize %zu <= outputCount %d\n", scaleSize, parm->common->outputCount);
        return nullptr;
    }

    parm->common->inputCount = inputCount;
    parm->bias.resize(parm->common->outputCount);
    if (nullptr != bias) {
        auto bptr = bias->readMap<float>();
        if (nullptr == bptr) {
            MNN_ERROR("_QConvolution1x1 quant: bias->readMap<float>() is nullptr!\n");
            return nullptr;
        }
        ::memcpy(parm->bias.data(), bptr, bias->getInfo()->size * sizeof(float));
    } else {
        ::memset(parm->bias.data(), 0, parm->bias.size() * sizeof(float));
    }
    if (nullptr != inputScale) {
        auto scale = inputScale->readMap<float>()[0];
        float zeroPoint = 0.0f;
        if (nullptr != inputZero) {
            zeroPoint = inputZero->readMap<int8_t>()[0];
        }
        input->writeScaleMap(scale, zeroPoint);
    }

    int n = parm->common->outputCount;
    int k = parm->common->inputCount;
    std::vector<int8_t> weightInt8(n * k);
    if (4 == weightBit) {
        int kDiv8 = k / 8;
        auto weightSrcInt8 = weight->readMap<int8_t>();
        for (int i = 0; i < kDiv8; ++i) {
            for (int u = 0; u < 4; ++u) {
                for (int v = 0; v < n; ++v) {
                    auto packed = weightSrcInt8[(i * 4 + u) * n + v];
                    int8_t item1 = packed >> 4;
                    int8_t item0 = packed - item1 * 16;
                    if (item0 >= 8) {
                        item0 -= 16;
                    }
                    MNN_ASSERT(item1 <= 7 && item1 >= -8);
                    weightInt8[v * k + i * 8 + u] = item0;
                    weightInt8[v * k + i * 8 + u + 4] = item1;
                }
            }
        }
    } else if (weightBit == 8) {
        ::memcpy(weightInt8.data(), weight->readMap<int8_t>(), n * k);
    } else {
        auto weightSrcUInt8 = weight->readMap<uint8_t>();
        auto weightUInt8 = (uint8_t*)weightInt8.data();
        _splitBufToArray(weightSrcUInt8, weightUInt8, n * k, weightBit);
        int offset = 1 << (weightBit - 1);
        for (int i = 0; i < n * k; ++i) {
            weightInt8[i] = (int)weightUInt8[i] - offset;
        }
    }

    int dstWeightBit = weightBit;
    if (8 == weightBit) {
        int maxV = -256;
        int minV = 256;
        for (int v = 0; v < n * k; ++v) {
            auto q = weightInt8[v];
            if (q > maxV) {
                maxV = q;
            }
            if (q < minV) {
                minV = q;
            }
        }
        int targetBit = 0;
        if (maxV >= 0) {
            targetBit = (int)ceil(log(maxV + 1) / log(2)) + 1;
        }
        if (minV < 0) {
            auto d1 = (int)ceil(log(-minV) / log(2)) + 1;
            if (d1 > targetBit) {
                targetBit = d1;
            }
        }
        dstWeightBit = targetBit;
    }
    if (dstWeightBit > 4) {
        dstWeightBit = 8;
    } else if (dstWeightBit > 1) {
        dstWeightBit = 4;
    } else {
        dstWeightBit = 1;
    }

    std::vector<float> scale;
    bool async = false;
    if (nullptr != wzeropoint) {
        if (_ReduceMax(_Abs(_Cast<float>(wzeropoint)))->readMap<float>()[0] >= 1e-11f) {
            async = true;
        }
    }
    if (async) {
        scale.resize(2 * scaleSize);
        if (wzeropoint->getInfo()->type.code == halide_type_float) {
            auto zeroPoint = wzeropoint->readMap<float>();
            auto scalePtr = wscale->readMap<float>();
            for (int i = 0; i < scaleSize; ++i) {
                scale[2 * i + 1] = scalePtr[i];
                scale[2 * i + 0] = zeroPoint[i];
            }
        } else {
            auto zeroPoint = wzeropoint->readMap<int8_t>();
            auto scalePtr = wscale->readMap<float>();
            for (int i = 0; i < scaleSize; ++i) {
                scale[2 * i + 1] = scalePtr[i];
                scale[2 * i + 0] = -scalePtr[i] * zeroPoint[i];
            }
        }
    } else {
        scale.resize(scaleSize);
        ::memcpy(scale.data(), wscale->readMap<float>(), scaleSize * sizeof(float));
    }
    auto kernelSize = n * k / scaleSize;
    parm->quanParameter = IDSTEncoder::encode(nullptr, scale, kernelSize, scaleSize, async, weightInt8.data(), 1,
                                              {dstWeightBit, false});
    return Variable::create(Expr::create(conv.get(), {input}));
}

static std::unique_ptr<OpT> _makeLayerNorm(const LayerNormInfo& info) {
    auto inputDim = info.hiddenSize;
    if (0 == inputDim) {
        if (nullptr != info.inputLayerNormWeight && nullptr != info.inputLayerNormWeight->getInfo()) {
            inputDim = (int)info.inputLayerNormWeight->getInfo()->size;
        } else {
            MNN_ERROR("_TransformerLayerNorm: hiddenSize is 0 and inputLayerNormWeight is missing!\n");
        }
    }
    std::unique_ptr<OpT> layerNorm(new OpT);
    layerNorm->type = OpType_LayerNorm;
    layerNorm->main.value = new LayerNormT;
    layerNorm->main.type = OpParameter_LayerNorm;
    layerNorm->main.AsLayerNorm()->axis = {-1};
    layerNorm->main.AsLayerNorm()->group = 1;
    layerNorm->main.AsLayerNorm()->epsilon = info.ln_eps;
    layerNorm->main.AsLayerNorm()->useRMSNorm = info.useRMSNorm;
    if (info.useC4) {
        layerNorm->defaultDimentionFormat = MNN_DATA_FORMAT_NC4HW4;
    }
    if (nullptr != info.inputLayerNormWeight) {
        layerNorm->main.AsLayerNorm()->beta.resize(inputDim);
        layerNorm->main.AsLayerNorm()->gamma.resize(inputDim);
        if (nullptr != info.inputLayerNormBias) {
            ::memcpy(layerNorm->main.AsLayerNorm()->beta.data(), info.inputLayerNormBias->readMap<float>(),
                     inputDim * sizeof(float));
        } else {
            ::memset(layerNorm->main.AsLayerNorm()->beta.data(), 0, inputDim * sizeof(float));
        }
        ::memcpy(layerNorm->main.AsLayerNorm()->gamma.data(), info.inputLayerNormWeight->readMap<float>(),
                 inputDim * sizeof(float));
    }
    return layerNorm;
}

std::pair<VARP, VARP> _BinaryLayerNorm(VARP r0, VARP r1, const LayerNormInfo& info) {
    std::unique_ptr<OpT> layerNorm = _makeLayerNorm(info);
    auto expr = Expr::create(layerNorm.get(), {r0, r1}, 2);
    return {Variable::create(expr, 0), Variable::create(expr, 1)};
}

VARP _TransformerLayerNorm(VARP hiddenState, const LayerNormInfo& info) {
    std::unique_ptr<OpT> layerNorm = _makeLayerNorm(info);
    return Variable::create(Expr::create(layerNorm.get(), {hiddenState}));
}

static void _fillRopeTable(float* dst, const std::vector<float>& cosTable, const std::vector<float>& sinTable,
                           int end, int halfDim) {
    const int tableSize = end * halfDim;
    for (int t = 0; t < end; ++t) {
        for (int i = 0; i < halfDim; ++i) {
            const int evenIndex = (2 * i) % halfDim;
            const int oddIndex = (2 * i + 1) % halfDim;
            const int srcEven = t * halfDim + evenIndex;
            const int srcOdd = t * halfDim + oddIndex;
            const int dstIndex = t * halfDim + i;
            dst[dstIndex] = cosTable[srcEven];
            dst[tableSize + dstIndex] = cosTable[srcOdd];
            dst[2 * tableSize + dstIndex] = sinTable[srcEven];
            dst[3 * tableSize + dstIndex] = sinTable[srcOdd];
        }
    }
}

VARP _PrecomputePosEmbedding(int dim, int end, float theta, bool interleaved) {
    if (dim % 2 != 0 || end <= 0 || theta <= 0.0f) {
        return nullptr;
    }

    const int halfDim = dim / 2;
    const int tableSize = end * halfDim;
    std::vector<float> cosTable(tableSize);
    std::vector<float> sinTable(tableSize);
    for (int t = 0; t < end; ++t) {
        for (int i = 0; i < halfDim; ++i) {
            const float exponent = static_cast<float>(2 * i) / static_cast<float>(dim);
            const float invFreq = 1.0f / std::pow(theta, exponent);
            const float freq = static_cast<float>(t) * invFreq;
            const int offset = t * halfDim + i;
            cosTable[offset] = std::cos(freq);
            sinTable[offset] = std::sin(freq);
        }
    }
    if (!interleaved) {
        std::vector<float> freqsCis(2 * tableSize);
        ::memcpy(freqsCis.data(), cosTable.data(), tableSize * sizeof(float));
        ::memcpy(freqsCis.data() + tableSize, sinTable.data(), tableSize * sizeof(float));
        auto res = _Const(freqsCis.data(), {2, end, halfDim}, NCHW, halide_type_of<float>());
        res.fix(VARP::CONSTANT);
        return res;
    }

    std::vector<float> ropeTables(4 * tableSize);
    _fillRopeTable(ropeTables.data(), cosTable, sinTable, end, halfDim);
    auto res = _Const(ropeTables.data(), {4, end, halfDim}, NCHW, halide_type_of<float>());
    res.fix(VARP::CONSTANT);
    return res;
}

VARPS _TransformerRoPE(VARP q, VARP k, VARP cosEven, VARP cosOdd, VARP sinEven, VARP sinOdd, const RopeInfo& info) {
    std::unique_ptr<OpT> qnorm;
    std::unique_ptr<OpT> knorm;
    if (nullptr != info.qNorm.inputLayerNormWeight.get()) {
        qnorm = _makeLayerNorm(info.qNorm);
    }
    if (nullptr != info.kNorm.inputLayerNormWeight.get()) {
        knorm = _makeLayerNorm(info.kNorm);
    }

    std::unique_ptr<OpT> ropeOp(new OpT);
    ropeOp->type = OpType_RoPE;
    ExtraT* extra = nullptr;
    if (info.cutHeadDim > 0 || nullptr != qnorm || nullptr != knorm) {
        ropeOp->main.type = OpParameter_Extra;
        extra = new ExtraT;
        extra->type = "RoPE";
        extra->engine = "MNN";
        ropeOp->main.value = extra;
    }
    if (nullptr != qnorm) {
        std::unique_ptr<AttributeT> attr(new AttributeT);
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(Op::Pack(builder, qnorm.get()));
        attr->key = "q_norm";
        attr->tensor.reset(new BlobT);
        attr->tensor->dataType = DataType_DT_INT8;
        attr->tensor->int8s.resize(builder.GetSize());
        ::memcpy(attr->tensor->int8s.data(), builder.GetBufferPointer(), builder.GetSize());
        extra->attr.emplace_back(std::move(attr));
    }
    if (nullptr != knorm) {
        std::unique_ptr<AttributeT> attr(new AttributeT);
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(Op::Pack(builder, knorm.get()));
        attr->key = "k_norm";
        attr->tensor.reset(new BlobT);
        attr->tensor->dataType = DataType_DT_INT8;
        attr->tensor->int8s.resize(builder.GetSize());
        ::memcpy(attr->tensor->int8s.data(), builder.GetBufferPointer(), builder.GetSize());
        extra->attr.emplace_back(std::move(attr));
    }
    if (info.cutHeadDim > 0) {
        std::unique_ptr<AttributeT> attr(new AttributeT);
        attr->key = "rope_cut_head_dim";
        attr->i = info.cutHeadDim;
        extra->attr.emplace_back(std::move(attr));
    }
    auto ropeExpr = Expr::create(ropeOp.get(), {q, k, cosEven, cosOdd, sinEven, sinOdd}, 2);
    return {Variable::create(ropeExpr, 0), Variable::create(ropeExpr, 1)};
}

} // namespace SafeTensorUtils
} // namespace Express
} // namespace MNN
