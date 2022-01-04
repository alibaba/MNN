#include "ConvDepthWiseExecution.hpp"
#include "core/ConvolutionCommon.hpp"
namespace MNN {
namespace CUDA {
struct constBuffer {
    int pad[2];
    int kernelSize[2];
    int stride[2];
    int dilate[2];
    int inputSize[2];
    int outputSize[2];
    int channel;
    int subChannel;
    int total;
    int activationType;
} uConstant;

ConvDepthWiseExecution::ConvDepthWiseExecution(const Op* op, Backend* bn) : Execution(bn) {
    mOp = op;
    auto pool = static_cast<CUDABackend*>(bn)->getStaticBufferPool();
    mConstBuffer = pool->alloc(sizeof(constBuffer));

    auto conv = mOp->main_as_Convolution2D();
    //weight host->device
    if(nullptr != conv->weight()) {
        int weightSize = conv->weight()->size();
        weightTensor.reset(Tensor::createDevice<float>({weightSize}));
        backend()->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
        mFilter = (void *)weightTensor.get()->buffer().device;
        cuda_check(cudaMemcpy(mFilter, conv->weight()->data(), conv->weight()->size()*sizeof(float), cudaMemcpyHostToDevice));

        mBias = nullptr;
        if(conv->bias()->size() != 0) {
            int biasSize = conv->bias()->size();
            biasTensor.reset(Tensor::createDevice<float>({biasSize}));
            backend()->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
            mBias = (void *)biasTensor.get()->buffer().device;
            cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
            use_bias_ = true;
        }
    }
}
ConvDepthWiseExecution::~ ConvDepthWiseExecution() {
    auto pool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    pool->free(mConstBuffer);
    if (nullptr != weightTensor) {
        backend()->onReleaseBuffer(weightTensor.get(), Backend::STATIC);
    }
    if(use_bias_ && nullptr != biasTensor) {
        backend()->onReleaseBuffer(biasTensor.get(), Backend::STATIC);
    }
}

ErrorCode ConvDepthWiseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto pad = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mOp->main_as_Convolution2D()->common());
    auto conv = mOp->main_as_Convolution2D();
    auto convCommon = mOp->main_as_Convolution2D()->common();
    constBuffer parameters;
    parameters.pad[0] = pad.first;
    parameters.pad[1] = pad.second;
    parameters.kernelSize[0] = convCommon->kernelX();
    parameters.kernelSize[1] = convCommon->kernelY();
    parameters.stride[0] = convCommon->strideX();
    parameters.stride[1] = convCommon->strideY();
    parameters.dilate[0] = convCommon->dilateX();
    parameters.dilate[1] = convCommon->dilateY();
    parameters.inputSize[0] = inputs[0]->width();
    parameters.inputSize[1] = inputs[0]->height();
    parameters.channel = inputs[0]->batch() * inputs[0]->channel();
    parameters.outputSize[0] = outputs[0]->width();
    parameters.outputSize[1] = outputs[0]->height();
    parameters.total = parameters.channel * parameters.outputSize[1] * parameters.outputSize[0];
    parameters.subChannel = inputs[0]->channel();
    parameters.activationType = convCommon->relu() ? 1 : (convCommon->relu6() ? 2 : 0);

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    runtime->memcpy((uint8_t*)mConstBuffer.first + mConstBuffer.second, &parameters, sizeof(constBuffer), MNNMemcpyHostToDevice);
    mTotalCount = parameters.total;

    return NO_ERROR;
}

__global__ void CONV_DW(const float* input, const float* kernel, const float* bias, float *output, const constBuffer* uConstant) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < uConstant->total; i += blockDim.x * gridDim.x) {
        {
            int iw = uConstant->inputSize[0];
            int ih = uConstant->inputSize[1];
            int c = uConstant->channel;
            int ow = uConstant->outputSize[0];
            int oh = uConstant->outputSize[1];
            int kw = uConstant->kernelSize[0];
            int kh = uConstant->kernelSize[1];
            int dw = uConstant->dilate[0];
            int dh = uConstant->dilate[1];
            int sw = uConstant->stride[0];
            int sh = uConstant->stride[1];
            int pw = uConstant->pad[0];
            int ph = uConstant->pad[1];
            int acttype = uConstant->activationType;

            int oz = i / (ow * oh);
            int tmp = i % (ow * oh);
            int oy = tmp / ow;
            int ox = tmp % ow;
            int kz = oz % uConstant->subChannel;
            
            int ix = ox * sw - pw;
            int iy = oy * sh - ph;
            float color = 0.0;
            if (bias != nullptr) {
                color = bias[kz];
            }

            int fx, fy, fz;
            for (fy=0; fy<kh; ++fy) {
                int sy = fy*dh + iy;
                if (sy >= ih || sy < 0) {
                    continue;
                }
                for (fx=0; fx<kw; ++fx) {
                    int sx = fx*dw + ix;
                    if (sx >= iw || sx < 0) {
                        continue;
                    }
                    float inputValue = input[0
                        + sx
                        + sy * iw
                        + oz * iw * ih
                    ];
                    float k = kernel[0
                        + fx
                        + fy * kw
                        + kz * kw * kh
                    ];
                    color  += k*inputValue;
                }
            }
            color = (acttype==1) ? max(0.0, color) : (acttype==2 ? (min(max(0.0, color), 6.0)) : color);
            output[0
                + ox
                + oy * ow
                + oz * ow * oh
            ] = color;
        }
    }
    return;
}


ErrorCode ConvDepthWiseExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto& prop = runtime->prop();
    int threads_num = prop.maxThreadsPerBlock;
    int block_num = prop.multiProcessorCount;
    auto constPtr = (uint8_t*)mConstBuffer.first + mConstBuffer.second;
    if (inputs.size() == 1) {
        CONV_DW<<<block_num, threads_num>>>((const float*)inputs[0]->deviceId(), (const float*)mFilter,
             (const float*)mBias, (float*)outputs[0]->deviceId(), (const constBuffer*)(constPtr));
    } else if (inputs.size() == 3) {
        CONV_DW<<<block_num, threads_num>>>((const float*)inputs[0]->deviceId(), (const float*)inputs[1]->deviceId(),
             (const float*)inputs[2]->deviceId(), (float*)outputs[0]->deviceId(), (const constBuffer*)constPtr);
    } else {
        MNN_ASSERT(inputs.size() == 2);
        CONV_DW<<<block_num, threads_num>>>((const float*)inputs[0]->deviceId(), (const float*)inputs[1]->deviceId(),
             nullptr, (float*)outputs[0]->deviceId(), (const constBuffer*)constPtr);
    }
    return NO_ERROR;
}



__global__ void DECONV_DW(const float* input, const float* kernel, const float* bias, float *output, const constBuffer* uConstant) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < uConstant->total; i += blockDim.x * gridDim.x) {
        {
            int iw = uConstant->inputSize[0];
            int ih = uConstant->inputSize[1];
            int c = uConstant->channel;
            int ow = uConstant->outputSize[0];
            int oh = uConstant->outputSize[1];
            int kw = uConstant->kernelSize[0];
            int kh = uConstant->kernelSize[1];
            int dw = uConstant->dilate[0];
            int dh = uConstant->dilate[1];
            int sw = uConstant->stride[0];
            int sh = uConstant->stride[1];
            int pw = uConstant->pad[0];
            int ph = uConstant->pad[1];

            int oz = i / (ow * oh);
            int tmp = i % (ow * oh);
            int oy = tmp / ow;
            int ox = tmp % ow;
            int kz = oz % uConstant->subChannel;
            
            int ix = ox + pw;
            int iy = oy + ph;
            float color = 0.0;
            if (bias != nullptr) {
                color = bias[kz];
            }

            int fx, fy, fz;
            for (fy=0; fy<kh; ++fy) {
                int sy = iy - fy*dh;
                int y = sy / sh;
                if (sy % sh == 0 && y >= 0 && y < ih) {
                    for (int fx=0; fx<kw; ++fx) {
                        int sx = ix - fx*dw;
                        int x = sx / sw;
                        if (sx % sw == 0 && x >= 0 && x < iw) {
                            float inputValue = input[0
                                + x
                                + y * iw
                                + oz * iw * ih
                            ];
                            float k = kernel[0
                                + fx
                                + fy * kw
                                + kz * kw * kh
                            ];
                            color  += k*inputValue;                            
                        }
                    }
                }
            }
            output[0
                + ox
                + oy * ow
                + oz * ow * oh
            ] = color;
        }
    }
    return;
}


ErrorCode DeconvDepthWiseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pad = ConvolutionCommon::convolutionTransposePad(inputs[0], outputs[0], convCommon);
    constBuffer parameters;
    parameters.pad[0] = pad.first;
    parameters.pad[1] = pad.second;
    parameters.kernelSize[0] = convCommon->kernelX();
    parameters.kernelSize[1] = convCommon->kernelY();
    parameters.stride[0] = convCommon->strideX();
    parameters.stride[1] = convCommon->strideY();
    parameters.dilate[0] = convCommon->dilateX();
    parameters.dilate[1] = convCommon->dilateY();
    parameters.inputSize[0] = inputs[0]->width();
    parameters.inputSize[1] = inputs[0]->height();
    parameters.channel = inputs[0]->batch() * inputs[0]->channel();
    parameters.outputSize[0] = outputs[0]->width();
    parameters.outputSize[1] = outputs[0]->height();
    parameters.total = parameters.channel * parameters.outputSize[1] * parameters.outputSize[0];
    parameters.subChannel = inputs[0]->channel();
    auto constPtr = (uint8_t*)mConstBuffer.first + mConstBuffer.second;

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    runtime->memcpy(constPtr, &parameters, sizeof(constBuffer), MNNMemcpyHostToDevice);
    mTotalCount = parameters.total;
    return NO_ERROR;
}

ErrorCode DeconvDepthWiseExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    int block_num = runtime->blocks_num(mTotalCount);
    int threads_num = runtime->threads_num();
    auto constPtr = (uint8_t*)mConstBuffer.first + mConstBuffer.second;
    if (inputs.size() > 2) {
        DECONV_DW<<<block_num, threads_num>>>((const float*)inputs[0]->deviceId(), (const float*)inputs[1]->deviceId(),
             (const float*)inputs[2]->deviceId(), (float*)outputs[0]->deviceId(), (const constBuffer*)constPtr);
    } else {
        DECONV_DW<<<block_num, threads_num>>>((const float*)inputs[0]->deviceId(), (const float*)inputs[1]->deviceId(),
             nullptr, (float*)outputs[0]->deviceId(), (const constBuffer*)constPtr);
    }
    return NO_ERROR;
}


class ConvDepthWiseExecutionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (OpType_ConvolutionDepthwise == op->type()) {
            return new ConvDepthWiseExecution(op, backend);
        }
        if (inputs.size() == 1) {
            MNN_PRINT("deconv depthwise not support 1 input yet\n");
            return nullptr;
        }
        return new DeconvDepthWiseExecution(op, backend);
    }
};

static CUDACreatorRegister<ConvDepthWiseExecutionCreator> __init(OpType_ConvolutionDepthwise);
static CUDACreatorRegister<ConvDepthWiseExecutionCreator> __init2(OpType_DeconvolutionDepthwise);
}
}