//
//  CPUUnary.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUUnary.hpp"
#include "UnaryUtils.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "compute/ConvOpt.h"
#include "compute/CommonOptFunction.h"
#include <MNN/AutoTime.hpp>
#include "math/Vec.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
using VecType = Math::Vec<int8_t, 8>;
CPUUnary::CPUUnary(Backend *b, MNNUnaryExecute proc, MNNUnaryExecuteInt8 procInt8, const Op* op) : MNN::Execution(b), mProc(proc), mProcInt8(procInt8){
    if (op->main_as_UnaryOp()->tableInt8()) {
        mTableBuffer.resize(255);
        ::memcpy(mTableBuffer.data(), op->main_as_UnaryOp()->tableInt8()->data(), 255);
    }
}

ErrorCode CPUUnary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());
    MNN_ASSERT(inputs[0]->getType() == halide_type_of<float>() || inputs[0]->getType() == halide_type_of<int32_t>());
    if (mProcInt8) {
        auto quantIn = TensorUtils::getDescribe(inputs[0])->quantAttr;
        auto quantOut = TensorUtils::getDescribe(outputs[0])->quantAttr;
        float outpScale = quantOut->scale == 0.f ? 0.f: 1.0f / quantOut->scale;
        mInpScale.push_back(quantIn->scale);
        mOupScale.push_back(outpScale);
        mInpZeroPoint.push_back(quantIn->zero);
        mOupZeroPoint.push_back(quantOut->zero);
        mMaxMinValue = {static_cast<ssize_t>(quantOut->min), static_cast<ssize_t>(quantOut->max)};
    }
    return NO_ERROR;
}

static void _Neg(void* out, const void* inp, int realSize) {
    MNNScaleAndAddBiasScalar((float*)out, (const float*)inp, 0.0f, -1.0f, realSize);
}
#ifdef MNN_USE_NEON
static inline void exeNegInt8 (int8_t* out, const int8_t* inp, int sizeQuad, int8x8_t inZeroPoint, int8x8_t outZeroPoint, float32x4_t inpScale, float32x4_t outScale) {
    for (int i = 0;i < sizeQuad; ++i) {
        int8x16_t negValue = vld1q_s8(inp);
        int16x8_t val16_0 = vmovl_s8(vget_low_s8(negValue));
        int16x8_t val16_1 = vmovl_s8(vget_high_s8(negValue));
        val16_0 = vsubw_s8(val16_0, inZeroPoint);
        val16_1 = vsubw_s8(val16_1, inZeroPoint);
        int32x4_t val32_00 = vmovl_s16(vget_low_s16(val16_0));
        int32x4_t val32_01 = vmovl_s16(vget_high_s16(val16_0));
        int32x4_t val32_10 = vmovl_s16(vget_low_s16(val16_1));
        int32x4_t val32_11 = vmovl_s16(vget_high_s16(val16_1));
        float32x4_t valF_00 = vcvtq_f32_s32(val32_00);
        float32x4_t valF_01 = vcvtq_f32_s32(val32_01);
        float32x4_t valF_10 = vcvtq_f32_s32(val32_10);
        float32x4_t valF_11 = vcvtq_f32_s32(val32_11);
        valF_00 = vmulq_f32(valF_00, inpScale);
        valF_01 = vmulq_f32(valF_01, inpScale);
        valF_10 = vmulq_f32(valF_10, inpScale);
        valF_11 = vmulq_f32(valF_11, inpScale);
        valF_00 = vnegq_f32(valF_00);
        valF_01 = vnegq_f32(valF_01);
        valF_10 = vnegq_f32(valF_10);
        valF_11 = vnegq_f32(valF_11);
        valF_00 = vmulq_f32(valF_00, outScale);
        valF_01 = vmulq_f32(valF_01, outScale);
        valF_10 = vmulq_f32(valF_10, outScale);
        valF_11 = vmulq_f32(valF_11, outScale);
        int32x4_t val_00 = vcvtq_s32_f32(valF_00);
        int32x4_t val_01 = vcvtq_s32_f32(valF_01);
        int32x4_t val_10 = vcvtq_s32_f32(valF_10);
        int32x4_t val_11 = vcvtq_s32_f32(valF_11);
        int16x4_t v16_0 = vqmovn_s32(val_00);
        int16x4_t v16_1 = vqmovn_s32(val_01);
        int16x4_t v16_2 = vqmovn_s32(val_10);
        int16x4_t v16_3 = vqmovn_s32(val_11);
        int16x8_t v16_4 = vcombine_s16(v16_0, v16_1);
        int16x8_t v16_5 = vcombine_s16(v16_2, v16_3);
        v16_4 = vaddw_s8(v16_4, outZeroPoint);
        v16_5 = vaddw_s8(v16_5, outZeroPoint);
        int8x8_t v8_0 = vqmovn_s16(v16_4);
        int8x8_t v8_1 = vqmovn_s16(v16_5);

        vst1_s8(out, v8_0);
        vst1_s8(out + 8, v8_1);
        inp  += 16;
        out += 16;
    }
}
#endif
static void _NegInt8(void* out, const void* inp, int realSize, QuanPrePostParameters* params) {
    int sizeDiv16 = realSize / 16;
    int remain = realSize % 16;
#ifdef MNN_USE_NEON
    int8_t* outPtr = (int8_t*)out;
    int8_t* inPtr  = (int8_t*)inp;
    int8x8_t inZeroPoint = vdup_n_s8(params->inputZeroPoint[0]);
    int8x8_t outZeroPoint = vdup_n_s8(params->outputZeroPoint[0]);
    float32x4_t inpScale = vdupq_n_f32(params->inputScale[0]);
    float32x4_t outScale = vdupq_n_f32(params->outputScale[0]);
    if (sizeDiv16 > 0) {
        exeNegInt8(outPtr, inPtr, sizeDiv16, inZeroPoint, outZeroPoint, inpScale, outScale);
    }
    if (remain > 0) {
        int8_t intmp[16] = {0};
        int8_t outmp[16] = {0};
        ::memcpy(intmp, reinterpret_cast<const int8_t*>(inp) + 16 * sizeDiv16, remain * sizeof(int8_t));
        exeNegInt8(outmp, intmp, 1, inZeroPoint, outZeroPoint, inpScale, outScale);
        ::memcpy(reinterpret_cast<int8_t*>(out) + 16 * sizeDiv16, outmp, remain * sizeof(int8_t));
    }
#else
#ifdef MNN_USE_SSE
    uint8_t* dst = (uint8_t*)out;
    uint8_t* src = (uint8_t*)inp;
    int offset = 128;
#else
    int8_t* dst = (int8_t*)out;
    int8_t* src = (int8_t*)inp;
    int offset = 0;
#endif
    int inzero_     = static_cast<int>(params->inputZeroPoint[0]);
    int outzero_    = static_cast<int>(params->outputZeroPoint[0]);
    float inscale_  = params->inputScale[0];
    float outscale_ = params->outputScale[0];
    int min_        = static_cast<int>(params->minValue);
    int max_        = static_cast<int>(params->maxValue);
    for (int i = 0; i < realSize; ++i) {
        int value = -(src[i] - inzero_ - offset) * inscale_ * outscale_ + outzero_;
        if (value > max_) {
            value = max_;
        }
        if (value < min_) {
            value = min_;
        }
        dst[i] = value + offset;
    }
#endif
}

static void _ABS(void* out, const void* inp, int realSize) {
    MNNReluWithSlopeCommon((float*)out, (const float*)inp, realSize, -1.0f);
}
#ifdef MNN_USE_NEON
static inline void exeAbsInt8(int8_t* out, const int8_t* inp, int sizeQuad, int8x8_t inZeroPoint, int8x8_t outZeroPoint, float32x4_t inpScale, float32x4_t outScale) {
    for (int i = 0;i < sizeQuad; ++i) {
        int8x16_t absValue = vld1q_s8(inp);
        int16x8_t val16_0 = vmovl_s8(vget_low_s8(absValue));
        int16x8_t val16_1 = vmovl_s8(vget_high_s8(absValue));
        val16_0 = vsubw_s8(val16_0, inZeroPoint);
        val16_1 = vsubw_s8(val16_1, inZeroPoint);
        int32x4_t val32_00 = vmovl_s16(vget_low_s16(val16_0));
        int32x4_t val32_01 = vmovl_s16(vget_high_s16(val16_0));
        int32x4_t val32_10 = vmovl_s16(vget_low_s16(val16_1));
        int32x4_t val32_11 = vmovl_s16(vget_high_s16(val16_1));
        float32x4_t valF_00 = vcvtq_f32_s32(val32_00);
        float32x4_t valF_01 = vcvtq_f32_s32(val32_01);
        float32x4_t valF_10 = vcvtq_f32_s32(val32_10);
        float32x4_t valF_11 = vcvtq_f32_s32(val32_11);
        valF_00 = vmulq_f32(valF_00, inpScale);
        valF_01 = vmulq_f32(valF_01, inpScale);
        valF_10 = vmulq_f32(valF_10, inpScale);
        valF_11 = vmulq_f32(valF_11, inpScale);
        valF_00 = vabsq_f32(valF_00);
        valF_01 = vabsq_f32(valF_01);
        valF_10 = vabsq_f32(valF_10);
        valF_11 = vabsq_f32(valF_11);
        valF_00 = vmulq_f32(valF_00, outScale);
        valF_01 = vmulq_f32(valF_01, outScale);
        valF_10 = vmulq_f32(valF_10, outScale);
        valF_11 = vmulq_f32(valF_11, outScale);
        int32x4_t val_00 = vcvtq_s32_f32(valF_00);
        int32x4_t val_01 = vcvtq_s32_f32(valF_01);
        int32x4_t val_10 = vcvtq_s32_f32(valF_10);
        int32x4_t val_11 = vcvtq_s32_f32(valF_11);
        int16x4_t v16_0 = vqmovn_s32(val_00);
        int16x4_t v16_1 = vqmovn_s32(val_01);
        int16x4_t v16_2 = vqmovn_s32(val_10);
        int16x4_t v16_3 = vqmovn_s32(val_11);
        int16x8_t v16_4 = vcombine_s16(v16_0, v16_1);
        int16x8_t v16_5 = vcombine_s16(v16_2, v16_3);
        v16_4 = vaddw_s8(v16_4, outZeroPoint);
        v16_5 = vaddw_s8(v16_5, outZeroPoint);
        int8x8_t v8_0 = vqmovn_s16(v16_4);
        int8x8_t v8_1 = vqmovn_s16(v16_5);

        vst1_s8(out, v8_0);
        vst1_s8(out + 8, v8_1);
        inp  += 16;
        out += 16;
    }
}
#endif
static void _ABSInt8(void* out, const void* inp, int realSize, QuanPrePostParameters* params) {
    int sizeDiv16 = realSize / 16;
    int remain = realSize % 16;
#ifdef MNN_USE_NEON
    int8_t* outPtr = (int8_t*)out;
    int8_t* inPtr  = (int8_t*)inp;
    int8x8_t inZeroPoint = vdup_n_s8(params->inputZeroPoint[0]);
    int8x8_t outZeroPoint = vdup_n_s8(params->outputZeroPoint[0]);
    float32x4_t inpScale = vdupq_n_f32(params->inputScale[0]);
    float32x4_t outScale = vdupq_n_f32(params->outputScale[0]);
    if (sizeDiv16 > 0) {
        exeAbsInt8(outPtr, inPtr, sizeDiv16, inZeroPoint, outZeroPoint, inpScale, outScale);
    }
    if (remain > 0) {
        int8_t intmp[16] = {0};
        int8_t outmp[16] = {0};
        ::memcpy(intmp, reinterpret_cast<const int8_t*>(inp) + 16 * sizeDiv16, remain * sizeof(int8_t));
        exeAbsInt8(outmp, intmp, 1, inZeroPoint, outZeroPoint, inpScale, outScale);
        ::memcpy(reinterpret_cast<int8_t*>(out) + 16 * sizeDiv16, outmp, remain * sizeof(int8_t));
    }
#else
#ifdef MNN_USE_SSE
    uint8_t* dst = (uint8_t*)out;
    uint8_t* src = (uint8_t*)inp;
    int offset = 128;
#else
    int8_t* dst = (int8_t*)out;
    int8_t* src = (int8_t*)inp;
    int offset = 0;
#endif
    int inzero_  = static_cast<int>(params->inputZeroPoint[0]);
    int outzero_ = static_cast<int>(params->outputZeroPoint[0]);
    for (int i = 0; i < realSize; ++i) {
        auto value = abs((src[i] - inzero_ - offset) * params->inputScale[0]);
        value = value * params->outputScale[0] + outzero_;
        if (value > params->maxValue) {
            value = params->maxValue;
        }
        if (value < params->minValue) {
            value = params->minValue;
        }
        dst[i] = value + offset;
    }
#endif
}
#ifdef MNN_USE_NEON
static inline void exeSignInt8 (int8_t* out, const int8_t* inp, int sizeQuad, int16x8_t one, int16x8_t negone, int16x8_t zero, int8x8_t inZeroPoint, int8x8_t outZeroPoint, float32x4_t outScale) {
        for (int i = 0;i < sizeQuad; ++i) {
            int8x16_t value = vld1q_s8(inp);
            int16x8_t vallow = vmovl_s8(vget_low_s8(value));
            int16x8_t valhi = vmovl_s8(vget_high_s8(value));
            vallow = vsubw_s8(vallow, inZeroPoint);
            valhi  = vsubw_s8(valhi, inZeroPoint);
            uint16x8_t lomask1  = vcgtq_s16(vallow, zero);
            uint16x8_t lomask_1 = vcltq_s16(vallow, zero);
            uint16x8_t himask1  = vcgtq_s16(valhi, zero);
            uint16x8_t himask_1 = vcltq_s16(valhi, zero);
            uint16x8_t zeromask_low = vceqq_u16(lomask1, lomask_1);
            uint16x8_t zeromask_hi = vceqq_u16(himask1, himask_1);
            vallow = vbslq_s16(lomask1, one, negone);
            vallow = vbslq_s16(zeromask_low, zero, vallow);
            valhi = vbslq_s16(himask1, one, negone);
            valhi = vbslq_s16(zeromask_hi, zero, valhi);
            int8x8_t v8_0 = vqmovn_s16(vallow);
            int8x8_t v8_1 = vqmovn_s16(valhi);
            vst1_s8(out, v8_0);
            vst1_s8(out + 8, v8_1);
            inp  += 16;
            out += 16;
        }
}
#endif
static void _SignInt8(void* out, const void* inp, int realSize, QuanPrePostParameters* params) {
    int sizeDiv16 = realSize / 16;
    int remain = realSize % 16;
#ifdef MNN_USE_NEON
    int8_t* outPtr = (int8_t*)out;
    int8_t* inPtr  = (int8_t*)inp;
    int16x8_t one = vdupq_n_s16(1);
    int16x8_t negone = vdupq_n_s16(-1);
    int16x8_t zero = vdupq_n_s16(0);
    int8x8_t inZeroPoint = vdup_n_s8(params->inputZeroPoint[0]);
    int8x8_t outZeroPoint = vdup_n_s8(params->outputZeroPoint[0]);
    float32x4_t outScale = vdupq_n_f32(params->outputScale[0]);
    if (sizeDiv16 > 0) {
        exeSignInt8(outPtr, inPtr, sizeDiv16, one, negone, zero, inZeroPoint, outZeroPoint, outScale);
    }
    if (remain > 0) {
        int8_t intmp[16] = {0};
        int8_t outmp[16] = {0};
        ::memcpy(intmp, reinterpret_cast<const int8_t*>(inp) + 16 * sizeDiv16, remain * sizeof(int8_t));
        exeSignInt8(outmp, intmp, 1, one, negone, zero, inZeroPoint, outZeroPoint, outScale);
        ::memcpy(reinterpret_cast<int8_t*>(out) + 16 * sizeDiv16, outmp, remain * sizeof(int8_t));
    }
#else
#ifdef MNN_USE_SSE
    uint8_t* dst = (uint8_t*)out;
    uint8_t* src = (uint8_t*)inp;
    int offset = 128;
#else
    int8_t* dst = (int8_t*)out;
    int8_t* src = (int8_t*)inp;
    int offset = 0;
#endif
    int inzero_  = static_cast<int>(params->inputZeroPoint[0]);
    int outzero_ = static_cast<int>(params->outputZeroPoint[0]);
    for (int i = 0; i < realSize; ++i) {
        auto value = src[i] - offset - inzero_;
        if (value > 0) {
            int f = 1 * params->outputScale[0] + outzero_;
            dst[i]     = f + offset;
        } else if (value < 0) {
            int f = -1 * params->outputScale[0] + outzero_;
            dst[i]     = f + offset;
        } else {
            dst[i] = outzero_ + offset;
        }
    }
#endif
}

static void _Square(void* out, const void* inp, int realSize) {
    MNNMatrixProdCommon((float*)out, (const float*)inp, (const float*)inp, realSize, 0, 0, 0, 1);
}

static void _EXP(void* outRaw, const void* inpRaw, int realSize) {
    auto out = (float*)outRaw;
    auto inp = (const float*)inpRaw;
    float offset[] = {
        1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(out, inp, offset, realSize);
}
static void _EXPM1(void* outRaw, const void* inpRaw, int realSize) {
    auto out = (float*)outRaw;
    auto inp = (const float*)inpRaw;
    float offset[] = {
        1.0f,
        -1.0f,
        0.0f,
        0.0f
    };
    MNNExp(out, inp, offset, realSize);
}

MNNUnaryExecute CPUUnary::selectForFloat(int type, int precision) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return _ABS;
        case UnaryOpOperation_SQUARE:
            return _Square;
        case UnaryOpOperation_NEG:
            return _Neg;
        case UnaryOpOperation_RSQRT:
            return _unaryOp<UnaryRsqrt<float>, float>;
        case UnaryOpOperation_EXP:
            return _EXP;
        case UnaryOpOperation_COS:
            return _unaryOp<UnaryCos<float>, float>;
        case UnaryOpOperation_SIN:
            return (MNNUnaryExecute)MNNSin;
        case UnaryOpOperation_SIGMOID:
            if (BackendConfig::Precision_Low == precision) {
                return (MNNUnaryExecute)MNNSigmoidLowp;
            } else {
                return (MNNUnaryExecute)MNNSigmoid;
            }
            break;
        case UnaryOpOperation_TANH:
            return (MNNUnaryExecute)MNNTanh;
        case UnaryOpOperation_TAN:
            return _unaryOp<UnaryTan<float>, float>;
        case UnaryOpOperation_ATAN:
            return _unaryOp<UnaryATan<float>, float>;
        case UnaryOpOperation_SQRT:
            return _unaryOp<UnarySqrt<float>, float>;
        case UnaryOpOperation_CEIL:
            return _unaryOp<UnaryCeil<float>, float>;
        case UnaryOpOperation_RECIPROCAL:
            return _unaryOp<UnaryRecipocal<float>, float>;
        case UnaryOpOperation_LOG1P:
            return _unaryOp<UnaryLog1p<float>, float>;
        case UnaryOpOperation_LOG:
            return _unaryOp<UnaryLog<float>, float>;
        case UnaryOpOperation_FLOOR:
            return _unaryOp<UnaryFloor<float>, float>;
        case UnaryOpOperation_BNLL:
            return _unaryOp<UnaryBNLL<float>, float>;
        case UnaryOpOperation_ACOSH:
            return _unaryOp<UnaryAcosh<float>, float>;
        case UnaryOpOperation_SINH:
            return _unaryOp<UnarySinh<float>, float>;
        case UnaryOpOperation_ASINH:
            return _unaryOp<UnaryAsinh<float>, float>;
        case UnaryOpOperation_ATANH:
            return _unaryOp<UnaryAtanh<float>, float>;
        case UnaryOpOperation_SIGN:
            return _unaryOp<UnarySign<float>, float>;
        case UnaryOpOperation_ROUND:
            return _unaryOp<UnaryRound<float>, float>;
        case UnaryOpOperation_COSH:
            return _unaryOp<UnaryCosh<float>, float>;
        case UnaryOpOperation_ERF:
            return _unaryOp<UnaryErf<float>, float>;
        case UnaryOpOperation_ERFC:
            return _unaryOp<UnaryErfc<float>, float>;
        case UnaryOpOperation_ERFINV:
            return _unaryOp<UnaryErfinv<float>, float>;
        case UnaryOpOperation_EXPM1:
            return _EXPM1;
        case UnaryOpOperation_ASIN:
            return _unaryOp<UnaryAsin<float>, float>;
        case UnaryOpOperation_ACOS:
            return _unaryOp<UnaryAcos<float>, float>;
        case UnaryOpOperation_HARDSWISH:
            return (MNNUnaryExecute)MNNHardSwishCommon;
        case UnaryOpOperation_GELU:
            return (MNNUnaryExecute)MNNGeluCommon;
        case UnaryOpOperation_GELU_STANDARD:
            return (MNNUnaryExecute)MNNGeluStandardCommon;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

static MNNUnaryExecute selectForInt(int type) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return _unaryOp<UnaryAbs<int32_t>, int32_t>;
        case UnaryOpOperation_NEG:
            return _unaryOp<UnaryNeg<int32_t>, int32_t>;
        case UnaryOpOperation_SQUARE:
            return _unaryOp<UnarySquare<int32_t>, int32_t>;
        case UnaryOpOperation_SIGN:
            return _unaryOp<UnarySign<int32_t>, int32_t>;
        default:
            break;
    }
    return nullptr;
}

MNNUnaryExecuteInt8 CPUUnary::selectForInt8(int type) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return _ABSInt8;
        case UnaryOpOperation_NEG:
            return _NegInt8;
        case UnaryOpOperation_SIGN:
            return _SignInt8;
        default:
            break;
    }
    return nullptr;
}
ErrorCode CPUUnary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto size = static_cast<CPUBackend*>(backend())->getTensorSize(input);
    auto schedule = ((CPUBackend*)backend())->multiThreadDivide(size);
    auto inputPtr = input->host<uint8_t>();
    auto outputPtr = output->host<uint8_t>();
    int outBytes = output->getType().bytes();
    if (halide_type_float == output->getType().code) {
        outBytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
    }
    if (mTableBuffer.data()) {
#ifdef MNN_USE_SSE
        uint8_t* srcO = inputPtr;
        uint8_t* dstO = outputPtr;
        int offset = 128;
#else
        int8_t* srcO = (int8_t*)inputPtr;
        int8_t* dstO = (int8_t*)outputPtr;
        int offset = 0;
#endif
        MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
            int start = schedule.first * (int)tId;
            int realSize = schedule.first;
            if (tId == schedule.second -1 ) {
                realSize = size - start;
            }
            if (realSize > 0) {
                auto inp = srcO + start;
                auto out = dstO + start;
                for (int i = 0; i < realSize; ++i) {
                    int idx = inp[i] - offset + 127;
                    out[i] = offset + mTableBuffer[idx];
                }
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
    if (mProcInt8) {
        MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
            QuanPrePostParameters params;
            params.inputScale = mInpScale.data();
            params.outputScale = mOupScale.data();
            params.inputZeroPoint= mInpZeroPoint.data();
            params.outputZeroPoint = mOupZeroPoint.data();
            params.maxValue = mMaxMinValue[1];
            params.minValue = mMaxMinValue[0];
            int start = schedule.first * (int)tId;
            int realSize = schedule.first;
            if (tId == schedule.second -1 ) {
                realSize = size - start;
            }
            if (realSize > 0) {
                auto inp = inputPtr + start;
                auto out = outputPtr + start;
                mProcInt8(out, inp, realSize, &params);
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
    MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
        int start = schedule.first * (int)tId;
        int realSize = schedule.first;
        if (tId == schedule.second -1 ) {
            realSize = size - start;
        }
        if (realSize > 0) {
            auto inp = inputPtr + start * outBytes;
            auto out = outputPtr + start * outBytes;
            mProc(out, inp, realSize);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class CPUUnaryCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto core = static_cast<CPUBackend*>(backend)->functions();
        auto precision = static_cast<CPUBackend*>(backend)->precisionMode();
        auto type = inputs[0]->getType();
        MNNUnaryExecute proc = nullptr;
        MNNUnaryExecuteInt8 procInt8 = nullptr;
        if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
            procInt8 = core->MNNSelectUnaryFunctionForInt8(op->main_as_UnaryOp()->opType());
        } else if (type.code == halide_type_int) {
            proc = selectForInt(op->main_as_UnaryOp()->opType());
        } else if (type.code == halide_type_float) {
            proc = core->MNNSelectUnaryFunctionForFloat(op->main_as_UnaryOp()->opType(), static_cast<CPUBackend*>(backend)->precisionMode());
        }
        if (nullptr == proc && nullptr == procInt8 && nullptr == op->main_as_UnaryOp()->tableInt8()) {
            MNN_ERROR("ERROR: Unary Op can not execute\n");
            return nullptr;
        }
        return new CPUUnary(backend, proc, procInt8, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUUnaryCreator, OpType_UnaryOp);

} // namespace MNN
