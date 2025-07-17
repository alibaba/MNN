#if MNN_KLEIDIAI_ENABLED
#include "KleidiAIDenseConvolution.hpp"

#include <numeric>

#include "CommonOptFunction.h"
#include "MNN/ErrorCode.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa.h"
#include "kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.h"
#include "kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme.h"
#include "kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.h"
#include "kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme.h"

namespace MNN {
template <typename T>
static void initWeight(const T* weight, const T* bias, T* cache, T* output, const std::vector<int>& shape,
                       const int bytes) {
    ::memset(cache, 0, sizeof(T) * std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
    ConvertOIHWToHWIO(cache, weight, shape);
    auto outputCount = shape[0];
    auto srcCount    = shape[1];
    auto kh          = shape[2];
    auto kw          = shape[3];
    if (bytes == 4) {
        kai_run_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(outputCount, kh * kw, srcCount, outputCount * sizeof(T),
                                                            cache, bias, output);
    } else if (bytes == 2) {
        kai_run_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(outputCount, kh * kw, srcCount, outputCount * sizeof(T),
                                                            cache, bias, output);
    } else {
        MNN_ERROR("Not fp32 and fp16, should not be called here\n");
        abort();
    }
}

KleidiAIDenseConvolution::KleidiAIDenseConvolution(const Convolution2DCommon* common, Backend* b,
                                                   const float* originWeight, size_t originWeightSize,
                                                   const float* bias, size_t biasSize,
                                                   std::shared_ptr<ConvolutionCommon::Int8Common> int8Info)
    : ConvolutionTiledExecutor(b, bias, biasSize) {
    auto outputCount = (int)biasSize;
    auto core        = static_cast<CPUBackend*>(b)->functions();
    int bytes        = core->bytes;
    auto srcCount    = (int)originWeightSize / outputCount / common->kernelX() / common->kernelY();
    if (core->matmulBytes != 0) {
        bytes = core->matmulBytes;
    }

    int kai_rhs_packed_size = 0;
    if (core->bytes == 4) {
        kai_rhs_packed_size = kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(
            outputCount, common->kernelY() * common->kernelX(), srcCount);
    } else if (core->bytes == 2) {
        kai_rhs_packed_size = kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
            outputCount, common->kernelY() * common->kernelX(), srcCount);
    } else {
        MNN_ERROR("Not fp32 and fp16, should not be called here\n");
        abort();
    }
    mResource->mWeight.reset(Tensor::createDevice<uint8_t>({kai_rhs_packed_size}));
    mResource->mBias.reset(Tensor::createDevice<uint8_t>({outputCount * core->bytes}));

    mValid = mValid && backend()->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    mValid = mValid && backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }

    std::shared_ptr<Tensor> cache(Tensor::createDevice<uint8_t>(
        {outputCount, srcCount * common->kernelX() * common->kernelY(), (int)sizeof(float)})); // cache must be float
    mValid = mValid && backend()->onAcquireBuffer(cache.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }

    std::vector<int> oihwShape = {outputCount, srcCount, common->kernelY(), common->kernelX()};
    if (core->bytes == 4) {
        MNN::initWeight(originWeight, bias, cache->host<float>(), mResource->mWeight->host<float>(), oihwShape,
                        core->bytes);
    } else if (core->bytes == 2) {
        for (int i = 0; i < outputCount; i++) {
            mResource->mBias->host<__fp16>()[i] = (__fp16)(bias[i]);
        }
        ConvertOIHWToHWIO(cache->host<__fp16>(), originWeight,
                          {outputCount, srcCount, common->kernelY(), common->kernelX()});
        kai_run_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
            outputCount, common->kernelY() * common->kernelX(), srcCount, outputCount * sizeof(__fp16),
            cache->host<__fp16>(), mResource->mBias->host<__fp16>(), mResource->mWeight->host<__fp16>());
    } else {
        MNN_ERROR("Not fp32 and fp16, should not be called here\n");
        abort();
    }

    backend()->onReleaseBuffer(cache.get(), Backend::STATIC);
    mProxy.reset(new KleidiAIDenseConvolutionImpl(common, b, mResource.get()));
}

KleidiAIDenseConvolution::KleidiAIDenseConvolution(std::shared_ptr<CPUConvolution::Resource> res,
                                                   const Convolution2DCommon* common, Backend* b)
    : ConvolutionTiledExecutor(res, b) {
    mProxy.reset(new KleidiAIDenseConvolutionImpl(common, b, mResource.get()));
}

KleidiAIDenseConvolution::~KleidiAIDenseConvolution() {
    // Do nothing
}

bool KleidiAIDenseConvolution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dense                     = new KleidiAIDenseConvolution(mResource, op->main_as_Convolution2D()->common(), bn);
    dense->mProxy->mConvPerfconfig = mProxy->mConvPerfconfig;
    *dst                           = dense;
    return true;
}

ErrorCode KleidiAIDenseConvolution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto code = mProxy->onExecute(mInputs, outputs);
    return code;
}
ErrorCode KleidiAIDenseConvolution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mInputs   = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
    auto code = mProxy->onResize(mInputs, outputs);
    if (NO_ERROR != code) {
        return code;
    }
    return NO_ERROR;
}

ErrorCode KleidiAIDenseConvolutionMultiInput::onExecute(const std::vector<Tensor*>& inputs,
                                                        const std::vector<Tensor*>& outputs) {
    auto function = static_cast<CPUBackend*>(backend())->functions();
    if (nullptr != mTempBias) {
        ::memset(mTempBias->host<float>(), 0, mTempBias->elementSize() * function->bytes);
        if (inputs.size() > 2) {
            ::memcpy(mTempBias->host<float>(), inputs[2]->host<float>(), inputs[2]->elementSize() * function->bytes);
        }
    }
    auto cache  = mTempWeightCache->host<float>();
    auto source = inputs[1]->host<float>();
    if (function->bytes == 4) {
        initWeight(source, mInputs[2]->host<float>(), cache, mTempWeight->host<float>(), inputs[1]->shape(),
                   function->bytes);
    } else if (function->bytes == 2) {
        initWeight(reinterpret_cast<const __fp16*>(source), mInputs[2]->host<__fp16>(),
                   reinterpret_cast<__fp16*>(cache), mTempWeight->host<__fp16>(), inputs[1]->shape(), function->bytes);
    } else {
        MNN_ERROR("Not fp32 and fp16, should not be called here\n");
        abort();
    }
    return mProxy->onExecute(mInputs, outputs);
}
ErrorCode KleidiAIDenseConvolutionMultiInput::onResize(const std::vector<Tensor*>& inputs,
                                                       const std::vector<Tensor*>& outputs) {
    int depth       = inputs[1]->channel();
    int outputCount = outputs[0]->channel();
    auto function   = static_cast<CPUBackend*>(backend())->functions();
    if (function->bytes == 4) {
        int kai_rhs_packed_size = kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(
            outputCount, inputs[1]->stride(1), depth);
        mTempWeight.reset(Tensor::createDevice<uint8_t>({kai_rhs_packed_size}));
    } else if (function->bytes == 2) {
        int kai_rhs_packed_size = kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
            outputCount, inputs[1]->stride(1), depth);
        mTempWeight.reset(Tensor::createDevice<uint8_t>({kai_rhs_packed_size}));
    } else {
        MNN_ERROR("Not fp32 and fp16, should not be called here\n");
        abort();
    }
    mTempWeightCache.reset(Tensor::createDevice<float>(
        {inputs[1]->height(), inputs[1]->width(), inputs[1]->channel(), inputs[1]->batch()}));
    auto res = backend()->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);
    res      = res && backend()->onAcquireBuffer(mTempWeightCache.get(), Backend::DYNAMIC);
    mTempBias.reset();
    if (!res) {
        return OUT_OF_MEMORY;
    }
    if (inputs.size() > 2 && inputs[2]->elementSize() % function->pack == 0) {
        mInputs = {inputs[0], mTempWeight.get(), inputs[2]};
    } else {
        mTempBias.reset(Tensor::createDevice<float>({UP_DIV(outputCount, function->pack) * function->pack}));
        backend()->onAcquireBuffer(mTempBias.get(), Backend::DYNAMIC);
        mInputs = {inputs[0], mTempWeight.get(), mTempBias.get()};
    }
    backend()->onReleaseBuffer(mTempWeightCache.get(), Backend::DYNAMIC);
    auto errorCode = mProxy->onResize(mInputs, outputs);
    backend()->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);
    if (nullptr != mTempBias) {
        backend()->onReleaseBuffer(mTempBias.get(), Backend::DYNAMIC);
    }
    return errorCode;
}

ErrorCode KleidiAIDenseConvolutionImpl::onResize(const std::vector<Tensor*>& inputs,
                                                 const std::vector<Tensor*>& outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto input   = inputs[0];
    auto weight  = inputs[1];
    Tensor* bias = nullptr;
    if (inputs.size() > 2) {
        bias = inputs[2];
    }
    auto core       = static_cast<CPUBackend*>(backend())->functions();
    int bytes       = core->bytes;
    int matmulBytes = bytes;
    if (core->matmulBytes != 0) {
        matmulBytes = core->matmulBytes;
    }
    auto ic     = input->channel();
    auto output = outputs[0];
    auto batch  = output->batch();

    auto outputChannel = output->channel();
    auto kernelSize    = mCommon->kernelX() * mCommon->kernelY();

    mTempBufferTranspose.buffer().type       = halide_type_of<uint8_t>();
    mTempBufferTranspose.buffer().dimensions = 1;
    int outputNhwSize                        = batch * output->height() * output->width();
    if (core->bytes == 4) {
        mTempBufferTranspose.buffer().dim[0].extent =
            kai_get_lhs_packed_size_lhs_imatmul_pack_x32p2vlx1_x32p_sme(outputNhwSize, kernelSize, ic);
    } else if (core->bytes == 2) {
        mTempBufferTranspose.buffer().dim[0].extent =
            kai_get_lhs_packed_size_lhs_imatmul_pack_x16p2vlx2_x16p_sme(outputNhwSize, kernelSize, ic);
    } else {
        MNN_ERROR("Not fp32 and fp16, should not be called here\n");
        abort();
    }
    TensorUtils::setLinearLayout(&mTempBufferTranspose);

    bool success = backend()->onAcquireBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    TensorUtils::getDescribe(&mOutputNHWC)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
    mOutputNHWC.buffer().dimensions                         = 4;
    mOutputNHWC.buffer().dim[0].extent                      = output->batch();
    mOutputNHWC.buffer().dim[1].extent                      = output->height();
    mOutputNHWC.buffer().dim[2].extent                      = output->width();
    mOutputNHWC.buffer().dim[3].extent                      = output->channel();
    mOutputNHWC.buffer().type                               = output->getType();
    success = backend()->onAcquireBuffer(&mOutputNHWC, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    TensorUtils::getDescribe(&mInputNHWC)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
    mInputNHWC.buffer().dimensions                         = 4;
    mInputNHWC.buffer().dim[0].extent                      = input->batch();
    mInputNHWC.buffer().dim[1].extent                      = input->height();
    mInputNHWC.buffer().dim[2].extent                      = input->width();
    mInputNHWC.buffer().dim[3].extent                      = input->channel();
    mInputNHWC.buffer().type                               = input->getType();
    success                                                = backend()->onAcquireBuffer(&mInputNHWC, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    TensorUtils::getDescribe(&mPadBuffer)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
    mPadBuffer.buffer().dimensions                         = 1;
    mPadBuffer.buffer().dim[0].extent                      = input->channel();
    mPadBuffer.buffer().type                               = input->getType();
    TensorUtils::setLinearLayout(&mPadBuffer);
    success = backend()->onAcquireBuffer(&mPadBuffer, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(&mOutputNHWC, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mInputNHWC, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mPadBuffer, Backend::DYNAMIC);

    backend()->onReleaseBuffer(&mTempBufferTranspose, Backend::DYNAMIC);

    auto postParameters = getPostParameters();
    mFunction.first     = ((CPUBackend*)backend())->threadNumber();

    auto padFull = ConvolutionCommon::convolutionPadFull(input, output, mCommon);
    ConvParams params{
        .inputChannel  = ic,
        .outputChannel = outputChannel,
        .kernelHeight  = mCommon->kernelY(),
        .kernelWidth   = mCommon->kernelX(),
        .strideHeight  = mCommon->strideY(),
        .strideWidth   = mCommon->strideX(),
        .padTop        = std::get<1>(padFull),
        .padBottom     = std::get<3>(padFull),
        .padLeft       = std::get<0>(padFull),
        .padRight      = std::get<2>(padFull),
        .dilatedHeight = mCommon->dilateY(),
        .dilatedWidth  = mCommon->dilateX(),
    };

    mFunction.second = [=](int tid) {
        // Convert NC4HW4 to NHWC
        auto inputShape = input->shape(); // TODO check for NC4HW4, should be the NCHW
        CPUTensorConverter::convert(input, &mInputNHWC, core);
        // Lhs packing
        if (bytes == 4) {
            int blockSize = kai_get_m_step_lhs_imatmul_pack_x32p2vlx1_x32p_sme();
            ::memset(mPadBuffer.host<float>(), 0, params.inputChannel * sizeof(float));
            auto table = IndirectionTable<float>(mInputNHWC.shape(), params, mInputNHWC.host<float>(),
                                                 mPadBuffer.host<float>(), blockSize);
            kai_run_lhs_imatmul_pack_x32p2vlx1_x32p_sme(outputNhwSize, kernelSize, ic, table.data.data(), 0,
                                                        mPadBuffer.host<uint8_t>(),
                                                        mTempBufferTranspose.host<uint8_t>());
        } else if (bytes == 2) {
            int blockSize = kai_get_m_step_lhs_imatmul_pack_x16p2vlx2_x16p_sme();
            ::memset(mPadBuffer.host<__fp16>(), 0, params.inputChannel * sizeof(__fp16));
            auto table = IndirectionTable<__fp16>(mInputNHWC.shape(), params, mInputNHWC.host<__fp16>(),
                                                  mPadBuffer.host<__fp16>(), blockSize);
            kai_run_lhs_imatmul_pack_x16p2vlx2_x16p_sme(outputNhwSize, kernelSize, ic, table.data.data(), 0,
                                                        mPadBuffer.host<uint8_t>(),
                                                        mTempBufferTranspose.host<uint8_t>());
        } else {
            MNN_ERROR("Not fp32 and fp16, should not be called here\n");
            abort();
        }

        // Run Matmul
        if (bytes == 4) {
            kai_run_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa(
                outputNhwSize, outputChannel, kernelSize, ic, mTempBufferTranspose.host<uint8_t>(),
                weight->host<uint8_t>(), mOutputNHWC.host<uint8_t>(), outputChannel * sizeof(float), postParameters[2],
                postParameters[3]);
        } else if (bytes == 2) {
            float max = postParameters[3] > 65504.f ? 65504.f : postParameters[3];
            kai_run_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(
                outputNhwSize, outputChannel, kernelSize, ic, mTempBufferTranspose.host<uint8_t>(),
                weight->host<uint8_t>(), mOutputNHWC.host<uint8_t>(), outputChannel * sizeof(__fp16), postParameters[2],
                max);
        } else {
            MNN_ERROR("Not fp32 and fp16, should not be called here\n");
            abort();
        }

        // Convert NHWC to NC4HW4
        CPUTensorConverter::convert(&mOutputNHWC, output, core);
    };
    return NO_ERROR;
}

ErrorCode KleidiAIDenseConvolutionImpl::onExecute(const std::vector<Tensor*>& inputs,
                                                  const std::vector<Tensor*>& outputs) {
    mFunction.second(0);
    return NO_ERROR;
}
} // namespace MNN
#endif
