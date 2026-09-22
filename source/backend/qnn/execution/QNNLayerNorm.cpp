//
//  QNNLayerNorm.cpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNLayerNorm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

QNNLayerNorm::QNNLayerNorm(Backend *backend, const Op *op, Tensor * input) : QNNCommonExecution(backend, op) {
    auto param = mOp->main_as_LayerNorm();

    mQnnDataType = mBackend->getUseFP16() ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;

    mInputDim = input->dimensions();

    mDimType = TensorUtils::getDimType(input);

    mEpsilon = param->epsilon();

    mUseRMSNorm = param->useRMSNorm();

    uint32_t axesSize = param->axis()->size();
    const int * axesData = param->axis()->data();
    int rawAxis = (axesData[0] >= 0) ? axesData[0] : (mInputDim + axesData[0]);
    mRealAxis = rawAxis;

    // set gamma and beta
    {
        bool hasGammaBeta = (param->gamma() && param->beta());
        mGammaBetaSize = 0;
        if (hasGammaBeta) {
            MNN_ASSERT(param->gamma()->size() == param->beta()->size());
            mGammaBetaSize = param->gamma()->size();
        }
        hasGammaBeta = hasGammaBeta || (param->external() && param->external()->size() > 1 && param->external()->data()[1] > 0);
        if (hasGammaBeta && mGammaBetaSize == 0) {
            mGammaBetaSize = param->external()->data()[1] / sizeof(float);
        }

        if(mGammaBetaSize > 0) {
            mGammaData.resize(mGammaBetaSize, 1.0f);
            mBetaData.resize(mGammaBetaSize);
            ::memcpy(mGammaData.data(), param->gamma()->data(), mGammaBetaSize * sizeof(float));
            ::memcpy(mBetaData.data(), param->beta()->data(), mGammaBetaSize * sizeof(float));
        }
    }

}

ErrorCode QNNLayerNorm::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::string nodeNameBase = "LayerNorm";
    nodeNameBase += "_";
    std::string inputTag = "I_";
    std::string outputTag = "O_";

    for (int i = 0; i < inputs.size(); i++) {
        inputTag += std::to_string(mBackend->getTensorIdx(inputs[i]));
        inputTag += "_";
    }

    for (int j = 0; j < outputs.size() - 1; j++) {
        outputTag += std::to_string(mBackend->getTensorIdx(outputs[j]));
        outputTag += "_";
    }
    outputTag += std::to_string(mBackend->getTensorIdx(outputs[outputs.size() - 1]));

    mNodeName = nodeNameBase + inputTag + outputTag;

    ErrorCode result = this->onEncode(inputs, outputs);
    if (result != NO_ERROR) {
        return result;
    }

    this->clean();

    return NO_ERROR;
}

void QNNLayerNorm::createGammaBeta(Qnn_DataType_t dataType){
    if(dataType == QNN_DATATYPE_FLOAT_16 || dataType == QNN_DATATYPE_FLOAT_32){
        this->createStaticFloatTensor("gamma", dataType, {(uint32_t) mGammaBetaSize}, mGammaData.data());                                      // mTempTensorWrappers[0], gamma
        this->createStaticFloatTensor("beta", dataType, {(uint32_t) mGammaBetaSize}, mBetaData.data());                                        // mTempTensorWrappers[1], beta
    }else{
        float minGamma = std::numeric_limits<float>::max();
        float maxGamma = -std::numeric_limits<float>::max();
        float minBeta = std::numeric_limits<float>::max();
        float maxBeta = -std::numeric_limits<float>::max();
        float gammaScale, betaScale;
        int gammaZeroPoint, betaZeroPoint;
        const bool useU8 = dataType == QNN_DATATYPE_UFIXED_POINT_8;
        float clampValue =
            useU8 ? 255.0f : (float)((1 << (16)) - 1);
        for(int i = 0; i < mGammaBetaSize; ++i){
            minGamma = std::min(minGamma, mGammaData[i]);
            maxGamma = std::max(maxGamma, mGammaData[i]);
        }
        for(int i = 0; i < mGammaBetaSize; ++i){
            minBeta = std::min(minBeta, mBetaData[i]);
            maxBeta = std::max(maxBeta, mBetaData[i]);
        }
        
        if(maxGamma - minGamma > 0.1f){
            gammaScale = (maxGamma - minGamma) / clampValue;
        }else{
            gammaScale = 0.1f / clampValue;
        }
        gammaZeroPoint = (int)roundf(minGamma/gammaScale);

        if(maxBeta - minBeta > 0.1f){
            betaScale = (maxBeta - minBeta) / clampValue;
        }else{
            betaScale = 0.1f / clampValue;
        }
        betaZeroPoint = (int)roundf(minBeta/betaScale);
        
        {
            Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS;
            Qnn_ScaleOffset_t tScaleOffsetEncoding;
            quantize.encodingDefinition = QNN_DEFINITION_DEFINED;
            quantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tScaleOffsetEncoding.scale = gammaScale;
            tScaleOffsetEncoding.offset = gammaZeroPoint;
            quantize.scaleOffsetEncoding = tScaleOffsetEncoding;
            
            if (useU8) {
                std::vector<uint8_t> gammaQuantData(mGammaBetaSize);
                for (int i = 0; i < mGammaBetaSize; ++i) {
                    gammaQuantData[i] = (uint8_t)roundf(
                        (mGammaData[i] - minGamma) / gammaScale);
                }
                this->createStaticTensor(
                    "gamma", QNN_DATATYPE_UFIXED_POINT_8,
                    {(uint32_t)mGammaBetaSize}, gammaQuantData.data(),
                    quantize);
            } else {
                std::vector<uint16_t> gammaQuantData(mGammaBetaSize);
                for (int i = 0; i < mGammaBetaSize; ++i) {
                    gammaQuantData[i] = (uint16_t)roundf(
                        (mGammaData[i] - minGamma) / gammaScale);
                }
                this->createStaticTensor(
                    "gamma", QNN_DATATYPE_UFIXED_POINT_16,
                    {(uint32_t)mGammaBetaSize}, gammaQuantData.data(),
                    quantize);
            }
        }
        
        {
            Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS;
            Qnn_ScaleOffset_t tScaleOffsetEncoding;
            quantize.encodingDefinition = QNN_DEFINITION_DEFINED;
            quantize.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            tScaleOffsetEncoding.scale = betaScale;
            tScaleOffsetEncoding.offset = betaZeroPoint;
            quantize.scaleOffsetEncoding = tScaleOffsetEncoding;
            
            if (useU8) {
                std::vector<uint8_t> betaQuantData(mGammaBetaSize);
                for (int i = 0; i < mGammaBetaSize; ++i) {
                    betaQuantData[i] = (uint8_t)roundf(
                        (mBetaData[i] - minBeta) / betaScale);
                }
                this->createStaticTensor(
                    "beta", QNN_DATATYPE_UFIXED_POINT_8,
                    {(uint32_t)mGammaBetaSize}, betaQuantData.data(),
                    quantize);
            } else {
                std::vector<uint16_t> betaQuantData(mGammaBetaSize);
                for (int i = 0; i < mGammaBetaSize; ++i) {
                    betaQuantData[i] = (uint16_t)roundf(
                        (mBetaData[i] - minBeta) / betaScale);
                }
                this->createStaticTensor(
                    "beta", QNN_DATATYPE_UFIXED_POINT_16,
                    {(uint32_t)mGammaBetaSize}, betaQuantData.data(),
                    quantize);
            }
        }
    }
}

ErrorCode QNNLayerNorm::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];

    std::vector<uint32_t> realInputShape = getNHWCShape(input);

    if (mGammaBetaSize == 0) {
        mGammaBetaSize = realInputShape[mRealAxis];
#ifdef QNN_VERBOSE
        MNN_PRINT("LayerNorm do not have original gamma beta, %d", mGammaBetaSize);
#endif
        mGammaData.resize(mGammaBetaSize, 1.0f);
        mBetaData.resize(mGammaBetaSize, 0.0f);
    } else {
        MNN_ASSERT(mGammaBetaSize == realInputShape[mRealAxis]);
    }

    Qnn_DataType_t dataType = mBackend->getNativeTensor(inputs[0])->v1.dataType;
    const bool fixedPointGraph =
        mBackend->requiresQuantizedGraph() &&
        dataType == QNN_DATATYPE_SFIXED_POINT_8;
    // Extra resources needed by Case Permute.
    bool needPermute = (mRealAxis == (mInputDim - 1)) ? false : true;
    if (fixedPointGraph) {
        // A fixed-point graph must never fall through to the native floating
        // LayerNorm/RmsNorm node. The integer contract currently normalizes a
        // contiguous last axis; reject any other layout instead of silently
        // partitioning it to CPU or FP16.
        if (needPermute) {
            MNN_ERROR(
                "MNN_INTEGER_LAYERNORM_AUDIT: backend=%s spec=1 "
                "runtime_integer=0 reason=NON_CONTIGUOUS_AXIS\n",
                mBackend->isDspBackend() ? "QUALCOMM_V66_DSP"
                                         : "QUALCOMM_V68_PLUS_HTP");
            return NOT_SUPPORT;
        }

#if QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR <= 27
        // V66 cannot execute the HTP S16 primitive LayerNorm graph directly.
        // Route both calibrated widths through one fused DSP node which
        // reproduces the HTP S16 stage scales and requantization boundaries.
        if (mBackend->isDspBackend() && !mBackend->v66LayerNormOpPackageName().empty() &&
            (realInputShape.back() == 256U ||
             realInputShape.back() == 8192U)) {
            const uint32_t normalizedSize = realInputShape.back();
            uint32_t normalizedSizeLog2 = 0;
            uint32_t sizeProbe = normalizedSize;
            while (sizeProbe > 1U && (sizeProbe & 1U) == 0U) {
                sizeProbe >>= 1U;
                ++normalizedSizeLog2;
            }
            if (sizeProbe != 1U ||
                !((normalizedSize == 256U && normalizedSizeLog2 == 8U) ||
                  (normalizedSize == 8192U && normalizedSizeLog2 == 13U))) {
                MNN_ERROR(
                    "MNN_INTEGER_LAYERNORM_AUDIT: backend=QUALCOMM_V66_DSP "
                    "spec=1 runtime_integer=0 reason=UNSUPPORTED_INNER "
                    "inner=%u\n", normalizedSize);
                return NOT_SUPPORT;
            }

            const auto inputQuantize =
                mBackend->getNativeTensor(inputs[0])->v1.quantizeParams;
            const auto outputQuantize =
                mBackend->getNativeTensor(outputs[0])->v1.quantizeParams;
            if (inputQuantize.encodingDefinition != QNN_DEFINITION_DEFINED ||
                outputQuantize.encodingDefinition != QNN_DEFINITION_DEFINED ||
                inputQuantize.quantizationEncoding !=
                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
                outputQuantize.quantizationEncoding !=
                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
                inputQuantize.scaleOffsetEncoding.offset != 0 ||
                outputQuantize.scaleOffsetEncoding.offset != 0 ||
                !(inputQuantize.scaleOffsetEncoding.scale > 0.0f) ||
                !(outputQuantize.scaleOffsetEncoding.scale > 0.0f)) {
                MNN_ERROR(
                    "MNN_INTEGER_LAYERNORM_AUDIT: backend=QUALCOMM_V66_DSP "
                    "spec=1 runtime_integer=0 reason=ASYMMETRIC_OR_INVALID_IO\n");
                return NOT_SUPPORT;
            }

            // QAIRT 2.36's DSP UDO bridge translates a directly connected
            // SFixed8 output to TF8 with a zero code of 0.  A LayerNorm
            // output is signed, so that convention clips every negative
            // result before the next graph node sees it.  Give the UDO an
            // explicitly centered UFixed8 boundary (zero code 128), then
            // let the native DSP Convert node restore the model's SFixed8
            // tensor without changing the represented real value.
            Qnn_QuantizeParams_t udoOutputQuantize = outputQuantize;
            udoOutputQuantize.scaleOffsetEncoding.offset = -128;
            const auto udoOutput = QNNTensorWrapper::create(
                mNodeName + "_udo_centered_u8_output",
                QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_UFIXED_POINT_8,
                realInputShape, udoOutputQuantize);
            mBackend->addTensor(udoOutput->getNativeTensor());
            mTempTensorWrappers.push_back(udoOutput);

            const double inputScale =
                inputQuantize.scaleOffsetEncoding.scale;
            const double outputScale =
                outputQuantize.scaleOffsetEncoding.scale;
            const double epsilonQ16Double =
                static_cast<double>(mEpsilon) /
                (inputScale * inputScale) * 65536.0;
            if (!std::isfinite(epsilonQ16Double) ||
                epsilonQ16Double < 0.0 ||
                epsilonQ16Double >
                    static_cast<double>(std::numeric_limits<uint32_t>::max())) {
                MNN_ERROR(
                    "MNN_INTEGER_LAYERNORM_AUDIT: backend=QUALCOMM_V66_DSP "
                    "spec=1 runtime_integer=0 reason=EPSILON_RANGE\n");
                return NOT_SUPPORT;
            }
            const uint32_t epsilonQ16 = static_cast<uint32_t>(
                std::max(1.0, std::round(epsilonQ16Double)));

            double gammaTensorMaximum = 0.0;
            double affineGammaMaximum = 1.0;
            double betaTensorMaximum = 0.0;
            for (const float gamma : mGammaData) {
                gammaTensorMaximum = std::max(
                    gammaTensorMaximum,
                    std::abs(static_cast<double>(gamma)));
                affineGammaMaximum = std::max(
                    affineGammaMaximum,
                    std::abs(static_cast<double>(gamma)));
            }
            for (const float beta : mBetaData) {
                betaTensorMaximum = std::max(
                    betaTensorMaximum,
                    std::abs(static_cast<double>(beta)));
            }
            const double normalizedScale = 16.0 / 32767.0;
            const double gammaScale = gammaTensorMaximum > 0.0
                ? gammaTensorMaximum / 32767.0
                : 1.0 / 32767.0;
            const double betaScale = betaTensorMaximum > 0.0
                ? betaTensorMaximum / 32767.0
                : 1.0 / 32767.0;
            const double affineMaximum =
                16.0 * affineGammaMaximum + betaTensorMaximum;
            const double affineStageScale = std::max(
                affineMaximum / 32767.0, 1.0e-12);
            const double normalizedGammaToAffine =
                normalizedScale * gammaScale / affineStageScale;
            const double affineToOutput = affineStageScale / outputScale;
            const double normalizedGammaToAffineQ30Double = std::ldexp(
                normalizedGammaToAffine, 30);
            const double affineToOutputQ30Double = std::ldexp(
                affineToOutput, 30);
            if (!std::isfinite(normalizedGammaToAffineQ30Double) ||
                !std::isfinite(affineToOutputQ30Double) ||
                normalizedGammaToAffineQ30Double < 1.0 ||
                affineToOutputQ30Double < 1.0 ||
                normalizedGammaToAffineQ30Double >
                    static_cast<double>(std::numeric_limits<uint32_t>::max()) ||
                affineToOutputQ30Double >
                    static_cast<double>(std::numeric_limits<uint32_t>::max())) {
                MNN_ERROR(
                    "MNN_INTEGER_LAYERNORM_AUDIT: backend=QUALCOMM_V66_DSP "
                    "spec=1 runtime_integer=0 reason=HTP_SCALE_RATIO_RANGE\n");
                return NOT_SUPPORT;
            }
            const uint32_t normalizedGammaToAffineQ30 =
                static_cast<uint32_t>(
                    std::round(normalizedGammaToAffineQ30Double));
            const uint32_t affineToOutputQ30 = static_cast<uint32_t>(
                std::round(affineToOutputQ30Double));

            std::vector<uint32_t> packedGammaMultiplier(
                (normalizedSize + 1U) / 2U, 0U);
            std::vector<int32_t> betaCode(normalizedSize);
            for (uint32_t index = 0U; index < normalizedSize; ++index) {
                const double gammaCode =
                    static_cast<double>(mGammaData[index]) / gammaScale;
                const int16_t gammaMultiplier = static_cast<int16_t>(std::max(
                    -32767.0, std::min(32767.0, std::round(gammaCode))));
                const uint32_t gammaBits =
                    static_cast<uint16_t>(gammaMultiplier);
                packedGammaMultiplier[index >> 1U] |=
                    gammaBits << ((index & 1U) * 16U);
                const double betaStaticCode = std::max(
                    -32767.0, std::min(
                        32767.0,
                        std::round(
                            static_cast<double>(mBetaData[index]) /
                            betaScale)));
                const double betaValue = std::round(
                    betaStaticCode * betaScale / affineStageScale);
                if (!std::isfinite(betaValue) ||
                    betaValue <
                        static_cast<double>(std::numeric_limits<int32_t>::min()) ||
                    betaValue >
                        static_cast<double>(std::numeric_limits<int32_t>::max())) {
                    MNN_ERROR(
                        "MNN_INTEGER_LAYERNORM_AUDIT: "
                        "backend=QUALCOMM_V66_DSP spec=1 runtime_integer=0 "
                        "reason=BETA_RANGE\n");
                    return NOT_SUPPORT;
                }
                betaCode[index] = static_cast<int32_t>(betaValue);
            }

            const uint32_t packedNormalizedSize =
                (normalizedSizeLog2 << 16U) | normalizedSize;
            const auto sizeParam = this->createParamScalar(
                "normalized_size", packedNormalizedSize);
            const auto epsilonParam = this->createParamScalar(
                "epsilon_q16", epsilonQ16);
            const auto rmsParam = this->createParamScalar(
                "use_rms_norm", mUseRMSNorm ? 1U : 0U);
            const auto normalizedGammaToAffineParam = this->createParamScalar(
                "normalized_gamma_to_affine_q30",
                normalizedGammaToAffineQ30);
            const auto affineToOutputParam = this->createParamScalar(
                "affine_to_output_q30", affineToOutputQ30);
            const auto gammaParam = this->createParamTensor(
                "gamma_multiplier", QNN_DATATYPE_UINT_32,
                {static_cast<uint32_t>(packedGammaMultiplier.size())},
                packedGammaMultiplier.data());
            const auto betaParam = this->createParamTensor(
                "beta_code", QNN_DATATYPE_INT_32,
                {normalizedSize}, betaCode.data());
            CLEAR_BEFORE_ADDING_NODE;
            mPackageName = mBackend->v66LayerNormOpPackageName();
            mNodeType = "DynamicInt8LayerNorm";
            mParams.push_back(*(sizeParam->getNativeParam()));
            mParams.push_back(*(epsilonParam->getNativeParam()));
            mParams.push_back(*(rmsParam->getNativeParam()));
            mParams.push_back(*(
                normalizedGammaToAffineParam->getNativeParam()));
            mParams.push_back(*(affineToOutputParam->getNativeParam()));
            mParams.push_back(*(gammaParam->getNativeParam()));
            mParams.push_back(*(betaParam->getNativeParam()));
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
            mOutputs.push_back(*(udoOutput->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(),
                mNodeType.c_str(), mParams, mInputs, mOutputs);
            {
                // Convert is accepted by host validation but is absent from
                // the V66 DSP graph finalizer.  Cast advertises the same
                // UFixed8 -> SFixed8 quantized kernel contract and performs
                // the required offset-aware requantization on this backend.
                CLEAR_BEFORE_ADDING_NODE;
                mPackageName = "qti.aisw";
                mNodeType = "Cast";
                mInputs.push_back(*(udoOutput->getNativeTensor()));
                mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
                mBackend->addNodeToGraph(
                    mOpConfigVersion,
                    (mNodeName + "_udo_u8_cast_to_model_s8").c_str(),
                    mPackageName.c_str(), mNodeType.c_str(), mParams,
                    mInputs, mOutputs);
            }
            return NO_ERROR;
        }
#endif

        // Implement dynamic LayerNorm entirely with fixed-point QNN
        // primitives. Unlike the former calibrated inverse-standard-
        // deviation vector, every invocation computes its own mean, variance
        // and reciprocal square root. MatMul with an integer vector of ones
        // expands [outer, 1] values to [outer, normalizedSize]; this also
        // avoids V66's unreliable dynamic broadcast path.
        auto makeQuantize = [](float scale) {
            Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS;
            quantize.encodingDefinition = QNN_DEFINITION_DEFINED;
            quantize.quantizationEncoding =
                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            quantize.scaleOffsetEncoding.scale =
                std::max(scale, 1.0e-12f);
            quantize.scaleOffsetEncoding.offset = 0;
            return quantize;
        };
        auto createNativeStage =
            [&](const std::string &name,
                const std::vector<uint32_t> &shape,
                const Qnn_QuantizeParams_t &quantize,
                Qnn_DataType_t stageType) {
                auto wrapper = QNNTensorWrapper::create(
                    mNodeName + "_" + name, QNN_TENSOR_TYPE_NATIVE,
                    stageType, shape, quantize);
                mBackend->addTensor(wrapper->getNativeTensor());
                mTempTensorWrappers.push_back(wrapper);
                return wrapper;
            };
#if QNN_API_VERSION_MAJOR == 2 && QNN_API_VERSION_MINOR <= 27
        // QAIRT 2.36's V66 DSP op package rejects S16 fixed-point inputs for
        // primitives such as ReduceMean. This restriction belongs to the V66
        // DSP execution domain, not to every backend compiled with the 2.36
        // headers. A V68+ HTP library may intentionally use those compatible
        // headers while loading a newer HTP runtime, and must retain S16.
        const bool kLegacyV66S8Intermediates = mBackend->isDspBackend();
        const Qnn_DataType_t kInternalDataType =
            kLegacyV66S8Intermediates
                ? QNN_DATATYPE_SFIXED_POINT_8
                : QNN_DATATYPE_SFIXED_POINT_16;
        const float kInternalQuantizedMaximum =
            kLegacyV66S8Intermediates ? 127.0f : 32767.0f;
        const uint32_t kLegacyNewtonIterations =
            kLegacyV66S8Intermediates ? 8U : 0U;
#else
        constexpr bool kLegacyV66S8Intermediates = false;
        constexpr Qnn_DataType_t kInternalDataType =
            QNN_DATATYPE_SFIXED_POINT_16;
        constexpr float kInternalQuantizedMaximum = 32767.0f;
        constexpr uint32_t kLegacyNewtonIterations = 0;
#endif
        auto createInternalStage =
            [&](const std::string &name,
                const std::vector<uint32_t> &shape,
                const Qnn_QuantizeParams_t &quantize) {
                return createNativeStage(
                    name, shape, quantize, kInternalDataType);
            };
        auto createInternalQuantizedTensor =
            [&](const std::string &name,
                const std::vector<float> &source,
                const std::vector<uint32_t> &shape) {
                float maxAbs = 0.0f;
                for (const float value : source) {
                    maxAbs = std::max(maxAbs, std::abs(value));
                }
                const float scale = maxAbs > 0.0f
                    ? maxAbs / kInternalQuantizedMaximum
                    : 1.0f / kInternalQuantizedMaximum;
                if (kLegacyV66S8Intermediates) {
                    std::vector<int8_t> quantized(source.size());
                    for (size_t index = 0; index < source.size(); ++index) {
                        const int32_t value = static_cast<int32_t>(
                            std::round(source[index] / scale));
                        quantized[index] = static_cast<int8_t>(
                            std::max<int32_t>(
                                -127, std::min<int32_t>(127, value)));
                    }
                    return this->createStaticTensor(
                        name, QNN_DATATYPE_SFIXED_POINT_8, shape,
                        quantized.data(), makeQuantize(scale));
                }
                std::vector<int16_t> quantized(source.size());
                for (size_t index = 0; index < source.size(); ++index) {
                    const int32_t value = static_cast<int32_t>(
                        std::round(source[index] / scale));
                    quantized[index] = static_cast<int16_t>(
                        std::max<int32_t>(
                            -32767, std::min<int32_t>(32767, value)));
                }
                return this->createStaticTensor(
                    name, QNN_DATATYPE_SFIXED_POINT_16, shape,
                    quantized.data(),
                    makeQuantize(scale));
            };
        auto createInternalQuantizedTensorWithScale =
            [&](const std::string &name,
                const std::vector<float> &source,
                const std::vector<uint32_t> &shape, float scale,
                bool preservePositive) {
                const float safeScale = std::max(scale, 1.0e-12f);
                if (kLegacyV66S8Intermediates) {
                    std::vector<int8_t> quantized(source.size());
                    for (size_t index = 0; index < source.size(); ++index) {
                        int32_t value = static_cast<int32_t>(
                            std::round(source[index] / safeScale));
                        if (preservePositive && source[index] > 0.0f) {
                            value = std::max<int32_t>(1, value);
                        }
                        quantized[index] = static_cast<int8_t>(
                            std::max<int32_t>(
                                -127, std::min<int32_t>(127, value)));
                    }
                    return this->createStaticTensor(
                        name, QNN_DATATYPE_SFIXED_POINT_8, shape,
                        quantized.data(), makeQuantize(safeScale));
                }
                std::vector<int16_t> quantized(source.size());
                for (size_t index = 0; index < source.size(); ++index) {
                    int32_t value = static_cast<int32_t>(
                        std::round(source[index] / safeScale));
                    if (preservePositive && source[index] > 0.0f) {
                        value = std::max<int32_t>(1, value);
                    }
                    quantized[index] = static_cast<int16_t>(
                        std::max<int32_t>(
                            -32767, std::min<int32_t>(32767, value)));
                }
                return this->createStaticTensor(
                    name, QNN_DATATYPE_SFIXED_POINT_16, shape,
                    quantized.data(), makeQuantize(safeScale));
            };

        uint32_t outer = 1;
        for (size_t index = 0; index + 1 < realInputShape.size(); ++index) {
            outer *= realInputShape[index];
        }
        const uint32_t normalizedSize = realInputShape.back();
        const std::vector<uint32_t> matrixShape = {outer, normalizedSize};
        const std::vector<uint32_t> reducedShape = {outer, 1};
        const auto inputQuantize =
            mBackend->getNativeTensor(inputs[0])->v1.quantizeParams;
        const auto outputQuantize =
            mBackend->getNativeTensor(outputs[0])->v1.quantizeParams;
        const bool v66Dsp = mBackend->isDspBackend();
        const float inputScale = std::max(
            inputQuantize.scaleOffsetEncoding.scale, 1.0e-12f);
        // Keep the graph boundary in INT8, but retain LayerNorm's dynamic
        // statistics in signed 16-bit fixed point. The effective magnitude is
        // capped at 32767 to leave multiplication headroom while providing
        // about eight more fractional bits than an S8 intermediate.
        const float inputMaximum = 127.0f * inputScale;
        const float wideInputScale = std::max(
            inputMaximum / kInternalQuantizedMaximum, 1.0e-12f);
        const float centeredMaximum =
            (mUseRMSNorm ? 1.0f : 2.0f) * inputMaximum;
        const float centeredScale = std::max(
            centeredMaximum / kInternalQuantizedMaximum, 1.0e-12f);
        const float squaredMaximum =
            centeredMaximum * centeredMaximum;
        const float squaredScale = std::max(
            squaredMaximum / kInternalQuantizedMaximum, 1.0e-12f);
        const float inverseStdMaximum =
            1.0f / std::sqrt(std::max(squaredScale, mEpsilon));
        const float inverseStdScale = std::max(
            inverseStdMaximum / kInternalQuantizedMaximum, 1.0e-12f);
        const float normalizedVarianceScale = 1.0f / 127.0f;
        const float newtonYScale = 16.0f / 127.0f;
        const float newtonYSquaredScale = 128.0f / 127.0f;
        const float newtonProductScale = 2.0f / 127.0f;
        const float newtonHalfProductScale = 1.0f / 127.0f;
        const float newtonCorrectionScale = 2.0f / 127.0f;
        const float normalizedMaximum = 16.0f;
        const float normalizedScale =
            normalizedMaximum / kInternalQuantizedMaximum;

        float maxAbsGamma = 1.0f;
        float maxAbsBeta = 0.0f;
        for (const float value : mGammaData) {
            maxAbsGamma = std::max(maxAbsGamma, std::abs(value));
        }
        for (const float value : mBetaData) {
            maxAbsBeta = std::max(maxAbsBeta, std::abs(value));
        }
        const float affineMaximum =
            normalizedMaximum * maxAbsGamma + maxAbsBeta;
        const float affineScale = std::max(
            affineMaximum / kInternalQuantizedMaximum, 1.0e-12f);

        const auto reshapedInput = createNativeStage(
            "int_matrix_input", matrixShape, inputQuantize,
            QNN_DATATYPE_SFIXED_POINT_8);
        const auto wideInput = createInternalStage(
            "wide_matrix_input", matrixShape,
            makeQuantize(wideInputScale));
        std::shared_ptr<QNNTensorWrapper> mean;
        std::shared_ptr<QNNTensorWrapper> expandedMean;
        std::shared_ptr<QNNTensorWrapper> centered;
        if (!mUseRMSNorm) {
            mean = createInternalStage(
                "int_mean", reducedShape, makeQuantize(wideInputScale));
            if (v66Dsp) {
                expandedMean = createInternalStage(
                    "int_mean_expanded", matrixShape,
                    makeQuantize(wideInputScale));
            }
            centered = createInternalStage(
                "int_centered", matrixShape, makeQuantize(centeredScale));
        }
        const auto squared = createInternalStage(
            "int_squared", matrixShape, makeQuantize(squaredScale));
        const auto variance = createInternalStage(
            "int_variance", reducedShape, makeQuantize(squaredScale));
        const auto stabilizedVariance = createInternalStage(
            "int_variance_epsilon", reducedShape,
            makeQuantize(squaredScale));
        const auto inverseStd = createInternalStage(
            "int_inverse_std", reducedShape,
            makeQuantize(inverseStdScale));
        std::shared_ptr<QNNTensorWrapper> normalizedVariance;
        std::vector<std::shared_ptr<QNNTensorWrapper>> newtonYSquared;
        std::vector<std::shared_ptr<QNNTensorWrapper>> newtonProduct;
        std::vector<std::shared_ptr<QNNTensorWrapper>> newtonHalfProduct;
        std::vector<std::shared_ptr<QNNTensorWrapper>> newtonCorrection;
        std::vector<std::shared_ptr<QNNTensorWrapper>> newtonY;
        if (kLegacyV66S8Intermediates) {
            normalizedVariance = createInternalStage(
                "int_variance_normalized", reducedShape,
                makeQuantize(normalizedVarianceScale));
            newtonYSquared.reserve(kLegacyNewtonIterations);
            newtonProduct.reserve(kLegacyNewtonIterations);
            newtonHalfProduct.reserve(kLegacyNewtonIterations);
            newtonCorrection.reserve(kLegacyNewtonIterations);
            newtonY.reserve(kLegacyNewtonIterations);
            for (uint32_t iteration = 0;
                 iteration < kLegacyNewtonIterations; ++iteration) {
                const std::string suffix = std::to_string(iteration);
                newtonYSquared.push_back(createInternalStage(
                    "int_newton_y_squared_" + suffix, reducedShape,
                    makeQuantize(newtonYSquaredScale)));
                newtonProduct.push_back(createInternalStage(
                    "int_newton_variance_y_squared_" + suffix,
                    reducedShape, makeQuantize(newtonProductScale)));
                newtonHalfProduct.push_back(createInternalStage(
                    "int_newton_half_product_" + suffix, reducedShape,
                    makeQuantize(newtonHalfProductScale)));
                newtonCorrection.push_back(createInternalStage(
                    "int_newton_correction_" + suffix, reducedShape,
                    makeQuantize(newtonCorrectionScale)));
                newtonY.push_back(createInternalStage(
                    "int_newton_y_" + suffix, reducedShape,
                    makeQuantize(newtonYScale)));
            }
        }
        std::shared_ptr<QNNTensorWrapper> expandedInverseStd;
        if (v66Dsp) {
            expandedInverseStd = createInternalStage(
                "int_inverse_std_expanded", matrixShape,
                makeQuantize(inverseStdScale));
        }
        const auto normalized = createInternalStage(
            "int_normalized", matrixShape,
            makeQuantize(normalizedScale));
        const auto affine = createInternalStage(
            "int_affine", matrixShape, makeQuantize(affineScale));
        const auto wideOutput = createInternalStage(
            "wide_matrix_output", matrixShape, makeQuantize(affineScale));
        const auto normalizedOutput = createNativeStage(
            "int_matrix_output", matrixShape, outputQuantize,
            QNN_DATATYPE_SFIXED_POINT_8);

        std::shared_ptr<QNNTensorWrapper> expandOnes;
        if (v66Dsp) {
            const std::vector<uint32_t> expandShape = {1, normalizedSize};
            const std::vector<float> onesData(normalizedSize, 1.0f);
            expandOnes = createInternalQuantizedTensorWithScale(
                "int_expand_ones", onesData, expandShape,
                1.0f / kInternalQuantizedMaximum, false);
        }

        std::vector<float> repeatedGamma(
            static_cast<size_t>(outer) * normalizedSize);
        std::vector<float> repeatedBeta(
            static_cast<size_t>(outer) * normalizedSize);
        for (uint32_t row = 0; row < outer; ++row) {
            std::copy(mGammaData.begin(), mGammaData.end(),
                      repeatedGamma.begin() +
                          static_cast<size_t>(row) * normalizedSize);
            std::copy(mBetaData.begin(), mBetaData.end(),
                      repeatedBeta.begin() +
                          static_cast<size_t>(row) * normalizedSize);
        }
        const auto layerGamma = createInternalQuantizedTensor(
            "int_layer_gamma", repeatedGamma, matrixShape);
        const auto layerBeta = createInternalQuantizedTensor(
            "int_layer_beta", repeatedBeta, matrixShape);
        const std::vector<float> epsilonData(outer, mEpsilon);
        const auto epsilonTensor = createInternalQuantizedTensorWithScale(
            "int_epsilon", epsilonData, reducedShape, squaredScale, true);
        std::shared_ptr<QNNTensorWrapper> inverseSquaredMaximumTensor;
        std::shared_ptr<QNNTensorWrapper> newtonHalfTensor;
        std::shared_ptr<QNNTensorWrapper> newtonThreeHalvesTensor;
        std::shared_ptr<QNNTensorWrapper> newtonInitialYTensor;
        std::shared_ptr<QNNTensorWrapper> inverseSqrtSquaredMaximumTensor;
        if (kLegacyV66S8Intermediates) {
            const std::vector<float> inverseSquaredMaximumData(
                outer, 1.0f / std::max(squaredMaximum, 1.0e-12f));
            const std::vector<float> newtonHalfData(outer, 0.5f);
            const std::vector<float> newtonThreeHalvesData(outer, 1.5f);
            const std::vector<float> newtonInitialYData(outer, 1.0f);
            const std::vector<float> inverseSqrtSquaredMaximumData(
                outer,
                1.0f / std::sqrt(std::max(squaredMaximum, 1.0e-12f)));
            inverseSquaredMaximumTensor = createInternalQuantizedTensor(
                "int_inverse_squared_maximum", inverseSquaredMaximumData,
                reducedShape);
            newtonHalfTensor = createInternalQuantizedTensor(
                "int_newton_half", newtonHalfData, reducedShape);
            newtonThreeHalvesTensor = createInternalQuantizedTensor(
                "int_newton_three_halves", newtonThreeHalvesData,
                reducedShape);
            newtonInitialYTensor = createInternalQuantizedTensor(
                "int_newton_initial_y", newtonInitialYData, reducedShape);
            inverseSqrtSquaredMaximumTensor = createInternalQuantizedTensor(
                "int_inverse_sqrt_squared_maximum",
                inverseSqrtSquaredMaximumData, reducedShape);
        }

        uint32_t reductionAxis = 1;
        const auto reduceAxes = this->createParamTensor(
            "axes", QNN_DATATYPE_UINT_32, {1}, &reductionAxis, "int");
        // The offline converter emits each param tensor once, and the QAIRT
        // QnnModel sample rejects a param tensor shared by two nodes. Give
        // the variance ReduceMean its own identical axes param tensor.
        const auto reduceAxesVariance = this->createParamTensor(
            "axes", QNN_DATATYPE_UINT_32, {1}, &reductionAxis,
            "int_variance");
        const auto keepDims =
            this->createParamScalar("keep_dims", true);
        std::shared_ptr<QNNParamScalarWrapper> transposeIn0;
        std::shared_ptr<QNNParamScalarWrapper> transposeIn1;
        if (v66Dsp) {
            transposeIn0 = this->createParamScalar("transpose_in0", false);
            transposeIn1 = this->createParamScalar("transpose_in1", false);
        }

        {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "Reshape";
            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
            mOutputs.push_back(*(reshapedInput->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_input_reshape").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            // Convert changes the fixed-point encoding while preserving the
            // represented real value. Cast would only widen the raw integer
            // code and would therefore apply the wrong S32 scale.
            mNodeType = "Convert";
            mInputs.push_back(*(reshapedInput->getNativeTensor()));
            mOutputs.push_back(*(wideInput->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_input_widen").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        if (!mUseRMSNorm) {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "ReduceMean";
            mInputs.push_back(*(wideInput->getNativeTensor()));
            mParams.push_back(*(reduceAxes->getNativeParam()));
            mParams.push_back(*(keepDims->getNativeParam()));
            mOutputs.push_back(*(mean->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_mean").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        if (!mUseRMSNorm && v66Dsp) {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "MatMul";
            mInputs.push_back(*(mean->getNativeTensor()));
            mInputs.push_back(*(expandOnes->getNativeTensor()));
            mParams.push_back(*(transposeIn0->getNativeParam()));
            mParams.push_back(*(transposeIn1->getNativeParam()));
            mOutputs.push_back(*(expandedMean->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_expand_mean").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        if (!mUseRMSNorm) {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "ElementWiseSubtract";
            mInputs.push_back(*(wideInput->getNativeTensor()));
            const auto meanInput = v66Dsp ? expandedMean : mean;
            mInputs.push_back(*(meanInput->getNativeTensor()));
            mOutputs.push_back(*(centered->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_center").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "ElementWiseMultiply";
            const auto normInput = mUseRMSNorm ? wideInput : centered;
            mInputs.push_back(*(normInput->getNativeTensor()));
            mInputs.push_back(*(normInput->getNativeTensor()));
            mOutputs.push_back(*(squared->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_square").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "ReduceMean";
            mInputs.push_back(*(squared->getNativeTensor()));
            mParams.push_back(*(reduceAxesVariance->getNativeParam()));
            mParams.push_back(*(keepDims->getNativeParam()));
            mOutputs.push_back(*(variance->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_variance").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "ElementWiseAdd";
            mInputs.push_back(*(variance->getNativeTensor()));
            mInputs.push_back(*(epsilonTensor->getNativeTensor()));
            mOutputs.push_back(*(stabilizedVariance->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion,
                (mNodeName + "_int_add_epsilon").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        if (kLegacyV66S8Intermediates) {
            // QAIRT 2.36 V66 accepts fixed-point Rsqrt/SquareRoot while
            // finalizing the graph but fails them during DSP execution. Scale
            // the dynamic variance into [0, 1], then evaluate reciprocal
            // square root with eight fixed-point Newton iterations:
            //   y[n+1] = y[n] * (1.5 - 0.5 * x * y[n]^2)
            // Starting at one is monotonic for x in (0, 1], and the epsilon
            // tensor guarantees a positive lower bound of one S8 code.
            {
                CLEAR_BEFORE_ADDING_NODE;
                mNodeType = "ElementWiseMultiply";
                mInputs.push_back(*(stabilizedVariance->getNativeTensor()));
                mInputs.push_back(*(
                    inverseSquaredMaximumTensor->getNativeTensor()));
                mOutputs.push_back(*(normalizedVariance->getNativeTensor()));
                mBackend->addNodeToGraph(
                    mOpConfigVersion,
                    (mNodeName + "_int_normalize_variance").c_str(),
                    mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                    mOutputs);
            }
            std::shared_ptr<QNNTensorWrapper> currentY =
                newtonInitialYTensor;
            for (uint32_t iteration = 0;
                 iteration < kLegacyNewtonIterations; ++iteration) {
                const std::string suffix = std::to_string(iteration);
                {
                    CLEAR_BEFORE_ADDING_NODE;
                    mNodeType = "ElementWiseMultiply";
                    mInputs.push_back(*(currentY->getNativeTensor()));
                    mInputs.push_back(*(currentY->getNativeTensor()));
                    mOutputs.push_back(*(
                        newtonYSquared[iteration]->getNativeTensor()));
                    mBackend->addNodeToGraph(
                        mOpConfigVersion,
                        (mNodeName + "_int_newton_y_squared_" + suffix).c_str(),
                        mPackageName.c_str(), mNodeType.c_str(), mParams,
                        mInputs, mOutputs);
                }
                {
                    CLEAR_BEFORE_ADDING_NODE;
                    mNodeType = "ElementWiseMultiply";
                    mInputs.push_back(*(
                        normalizedVariance->getNativeTensor()));
                    mInputs.push_back(*(
                        newtonYSquared[iteration]->getNativeTensor()));
                    mOutputs.push_back(*(
                        newtonProduct[iteration]->getNativeTensor()));
                    mBackend->addNodeToGraph(
                        mOpConfigVersion,
                        (mNodeName + "_int_newton_product_" + suffix).c_str(),
                        mPackageName.c_str(), mNodeType.c_str(), mParams,
                        mInputs, mOutputs);
                }
                {
                    CLEAR_BEFORE_ADDING_NODE;
                    mNodeType = "ElementWiseMultiply";
                    mInputs.push_back(*(
                        newtonProduct[iteration]->getNativeTensor()));
                    mInputs.push_back(*(newtonHalfTensor->getNativeTensor()));
                    mOutputs.push_back(*(
                        newtonHalfProduct[iteration]->getNativeTensor()));
                    mBackend->addNodeToGraph(
                        mOpConfigVersion,
                        (mNodeName + "_int_newton_half_" + suffix).c_str(),
                        mPackageName.c_str(), mNodeType.c_str(), mParams,
                        mInputs, mOutputs);
                }
                {
                    CLEAR_BEFORE_ADDING_NODE;
                    mNodeType = "ElementWiseSubtract";
                    mInputs.push_back(*(
                        newtonThreeHalvesTensor->getNativeTensor()));
                    mInputs.push_back(*(
                        newtonHalfProduct[iteration]->getNativeTensor()));
                    mOutputs.push_back(*(
                        newtonCorrection[iteration]->getNativeTensor()));
                    mBackend->addNodeToGraph(
                        mOpConfigVersion,
                        (mNodeName + "_int_newton_correction_" + suffix).c_str(),
                        mPackageName.c_str(), mNodeType.c_str(), mParams,
                        mInputs, mOutputs);
                }
                {
                    CLEAR_BEFORE_ADDING_NODE;
                    mNodeType = "ElementWiseMultiply";
                    mInputs.push_back(*(currentY->getNativeTensor()));
                    mInputs.push_back(*(
                        newtonCorrection[iteration]->getNativeTensor()));
                    mOutputs.push_back(*(
                        newtonY[iteration]->getNativeTensor()));
                    mBackend->addNodeToGraph(
                        mOpConfigVersion,
                        (mNodeName + "_int_newton_y_" + suffix).c_str(),
                        mPackageName.c_str(), mNodeType.c_str(), mParams,
                        mInputs, mOutputs);
                }
                currentY = newtonY[iteration];
            }
            {
                CLEAR_BEFORE_ADDING_NODE;
                mNodeType = "ElementWiseMultiply";
                mInputs.push_back(*(currentY->getNativeTensor()));
                mInputs.push_back(*(
                    inverseSqrtSquaredMaximumTensor->getNativeTensor()));
                mOutputs.push_back(*(inverseStd->getNativeTensor()));
                mBackend->addNodeToGraph(
                    mOpConfigVersion,
                    (mNodeName + "_int_newton_inverse_std").c_str(),
                    mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                    mOutputs);
            }
        } else {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "ElementWiseRsqrt";
            mInputs.push_back(*(stabilizedVariance->getNativeTensor()));
            mOutputs.push_back(*(inverseStd->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_rsqrt").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        if (v66Dsp) {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "MatMul";
            mInputs.push_back(*(inverseStd->getNativeTensor()));
            mInputs.push_back(*(expandOnes->getNativeTensor()));
            mParams.push_back(*(transposeIn0->getNativeParam()));
            mParams.push_back(*(transposeIn1->getNativeParam()));
            mOutputs.push_back(*(expandedInverseStd->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion,
                (mNodeName + "_int_expand_inverse_std").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "ElementWiseMultiply";
            const auto normInput = mUseRMSNorm ? wideInput : centered;
            mInputs.push_back(*(normInput->getNativeTensor()));
            const auto inverseInput =
                v66Dsp ? expandedInverseStd : inverseStd;
            mInputs.push_back(*(inverseInput->getNativeTensor()));
            mOutputs.push_back(*(normalized->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_normalize").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "ElementWiseMultiply";
            mInputs.push_back(*(normalized->getNativeTensor()));
            mInputs.push_back(*(layerGamma->getNativeTensor()));
            mOutputs.push_back(*(affine->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_gamma").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "ElementWiseAdd";
            mInputs.push_back(*(affine->getNativeTensor()));
            mInputs.push_back(*(layerBeta->getNativeTensor()));
            mOutputs.push_back(*(wideOutput->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_beta").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "Convert";
            mInputs.push_back(*(wideOutput->getNativeTensor()));
            mOutputs.push_back(*(normalizedOutput->getNativeTensor()));
            mBackend->addNodeToGraph(
                mOpConfigVersion,
                (mNodeName + "_int_output_requantize_once").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        {
            CLEAR_BEFORE_ADDING_NODE;
            mNodeType = "Reshape";
            mInputs.push_back(*(normalizedOutput->getNativeTensor()));
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
            mBackend->addNodeToGraph(
                mOpConfigVersion, (mNodeName + "_int_output_reshape").c_str(),
                mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs,
                mOutputs);
        }
        return NO_ERROR;
    }
    // Create resources for the native floating-point LayerNorm path. Integer
    // paths create only the parameters they actually consume.
    this->createParamScalar("epsilon", mEpsilon); // mParamScalarWrappers[0]
    uint32_t tempPtr[1] = {(uint32_t) mInputDim - 1};
    this->createParamTensor(
        "axes", QNN_DATATYPE_UINT_32, {1}, (void *)tempPtr);
    createGammaBeta(dataType);
    if (needPermute) {
        std::vector<uint32_t> realInputShape = getNHWCShape(inputs[0]);

        std::vector<uint32_t> permData(mInputDim, 0);
        std::vector<uint32_t> tempInputOutputShape(mInputDim, 0);

        for (int i = 0; i < mRealAxis; i++) {
            permData[i] = i;
            tempInputOutputShape[i] = realInputShape[i];
        }
        permData[mRealAxis] = mInputDim - 1;
        tempInputOutputShape[mRealAxis] = realInputShape[mInputDim - 1];
        for (int j = mRealAxis + 1; j < mInputDim - 1; j++) {
            permData[j] = j;
            tempInputOutputShape[j] = realInputShape[j];
        }
        permData[mInputDim - 1] = mRealAxis;
        tempInputOutputShape[mInputDim - 1] = realInputShape[mRealAxis];

        #ifdef QNN_VERBOSE
        MNN_PRINT("QNN LayerNorm Permute data:");
        for(int i = 0; i < permData.size(); i++) {
            MNN_PRINT("%d ", permData[i]);
        }
        MNN_PRINT("\n");
        MNN_PRINT("QNN LayerNorm tempShape data:");
        for(int i = 0; i < tempInputOutputShape.size(); i++) {
            MNN_PRINT("%d ", tempInputOutputShape[i]);
        }
        MNN_PRINT("\n");
        #endif

        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) mInputDim}, (void *) permData.data(), "before");           // mParamTensorWrappers[1], perm before
        this->createParamTensor("perm", QNN_DATATYPE_UINT_32, {(uint32_t) mInputDim}, (void *) permData.data(), "after");            // mParamTensorWrappers[2], perm after
        this->createStageTensor("tempInput", mQnnDataType, tempInputOutputShape);                                                       // mTempTensorWrappers[2], temp input
        this->createStageTensor("tempOutput", mQnnDataType, tempInputOutputShape);                                                      // mTempTensorWrappers[3], temp output
    }


    #ifdef QNN_VERBOSE
    MNN_PRINT("QNN LayerNorm useFp16:%d \ninput0:", mBackend->getUseFP16());
    auto shape0 = inputs[0]->shape();
    for(int i = 0; i < shape0.size(); i++) {
        MNN_PRINT("%d x ", shape0[i]);
    }
    MNN_PRINT("\noutput:");
    auto outShape = outputs[0]->shape();
    for(int i = 0; i < outShape.size(); i++) {
        MNN_PRINT("%d x ", outShape[i]);
    }
    MNN_PRINT("\n");
    MNN_PRINT("need Permute:%d, gamma:%d, reduceAxis:%d,\n", needPermute, mGammaBetaSize, mRealAxis);

    int rank = inputs.at(0)->dimensions();
    for(int i = 0; i < rank; i++) {
        MNN_PRINT("%d ", inputs.at(0)->length(i));
    }
    #endif

    // Add Nodes to Graph.
    if (needPermute) {
        return this->onEncodeNormWithPermute(inputs, outputs);
    }

    #ifdef QNN_LAYERNORM_RESHAPE_3D
    if(mInputDim == 4)
    {
        uint32_t tempPtr[1] = {(uint32_t)2}; // Qnn only allows the last dim for norm.
        this->createParamTensor("axes", QNN_DATATYPE_UINT_32, {1}, (void*)tempPtr, "redefine");
        this->createStageTensor("InputReshapeTensor", dataType,
                                std::vector<int>({inputs[0]->length(0), inputs[0]->length(2) * inputs[0]->length(3),
                                                  inputs[0]->length(1)}));
        this->createStageTensor("OutputReshapeTensor", dataType,
                                std::vector<int>({inputs[0]->length(0), inputs[0]->length(2) * inputs[0]->length(3),
                                                  inputs[0]->length(1)}));
        // reshape input
        {
            std::string name = mNodeName + "_input_reshape";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType = "Reshape";

            mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // input0
            mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // temp input
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }

        {
            std::string name = mNodeName + "_norm";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType = mUseRMSNorm ? "RmsNorm" : "LayerNorm";

            mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor()));
            mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // gamma
            mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // beta

            mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // eps
            mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // axes

            mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor()));

            mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        // reshape output
        {
            std::string name = mNodeName + "_output_reshape";
            mParams.clear();
            mInputs.clear();
            mOutputs.clear();
            mNodeType = "Reshape";

            mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // temp output
            mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // input0
            mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
        }
        return NO_ERROR;
    }
    #endif

    mNodeType = mUseRMSNorm ? "RmsNorm" : "LayerNorm";

    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // gamma
    mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // beta

    mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // eps
    mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // axes

    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);

    return NO_ERROR;
}

ErrorCode QNNLayerNorm::onEncodeNormWithPermute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // Permute before norm.
    {
        mNodeType.clear();
        mInputs.clear();
        mParams.clear();
        mOutputs.clear();

        std::string name = mNodeName + "_before";
        mNodeType = "Transpose";
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0]))); // inputs[0]
        mParams.push_back(*(mParamTensorWrappers[1]->getNativeParam())); // perm before
        mOutputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // temp input

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Norm.
    {
        std::string name = mNodeName + "_norm";
        mNodeType.clear();
        mInputs.clear();
        mParams.clear();
        mOutputs.clear();

        mNodeType = mUseRMSNorm ? "RmsNorm" : "LayerNorm";
        mInputs.push_back(*(mTempTensorWrappers[2]->getNativeTensor())); // temp input
        mInputs.push_back(*(mTempTensorWrappers[0]->getNativeTensor())); // gamma
        mInputs.push_back(*(mTempTensorWrappers[1]->getNativeTensor())); // beta

        mParams.push_back(*(mParamScalarWrappers[0]->getNativeParam())); // eps
        mParams.push_back(*(mParamTensorWrappers[0]->getNativeParam())); // axes
    
        mOutputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // temp output
    
        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    // Permute after norm.
    {
        mNodeType.clear();
        mInputs.clear();
        mParams.clear();
        mOutputs.clear();

        std::string name = mNodeName + "_after";
        mNodeType = "Transpose";
        mInputs.push_back(*(mTempTensorWrappers[3]->getNativeTensor())); // temp output
        mParams.push_back(*(mParamTensorWrappers[2]->getNativeParam())); // perm after
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0]))); // outputs[0]

        mBackend->addNodeToGraph(mOpConfigVersion, name.c_str(), mPackageName.c_str(), mNodeType.c_str(), mParams, mInputs, mOutputs);
    }

    return NO_ERROR;
}

class QNNLayerNormCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution * onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* backend) const override {
        auto inputDim = inputs[0]->dimensions();
        if (inputDim > 4) {
            return nullptr;
        }

        auto param = op->main_as_LayerNorm();

        if (param->group() > 1) {
            return nullptr;
        }

        if (param->axis()->size() != 1) {
            return nullptr;
        }

        return new QNNLayerNorm(backend, op, inputs[0]);
    }
};

REGISTER_QNN_OP_CREATOR(QNNLayerNormCreator, OpType_LayerNorm)
#endif
} // end namespace MNN
} // namespace MNN
