#include "ConvDepthWiseExecution.hpp"
#include "core/ConvolutionCommon.hpp"
#include "Raster.cuh"
#include <float.h>
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

template<typename T>
__global__ void CONV_DW(const T* input, 
    const half* kernel, 
    const half* bias, 
    T *output, 
    const constBuffer* uConstant
) {
    float maxV = uConstant->maxValue;
    float minV = uConstant->minValue;
    int iw = uConstant->inputSize[0];
    int ih = uConstant->inputSize[1];
    int c = uConstant->channel;
    int c_p = c * PACK_NUMBER;
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

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < uConstant->total; index += blockDim.x * gridDim.x) {
        int i = index / c_p;
        int oz = index % c_p;
        int ob = i / (ow * oh);
        int tmp = i % (ow * oh);
        int oy = tmp / ow;
        int ox = tmp % ow;
        
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color = bias[oz];
        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                float inp = input[0
                    + sx * c_p
                    + sy * iw * c_p
                    + ob * iw * ih * c_p
                    + oz
                ];
                float ker = kernel[0
                    + fx
                    + fy * kw
                    + oz * kw * kh
                ];
                color = color + inp * ker;
            }
        }
        color = max(color, minV);
        color = min(color, maxV);

        output[0
            + ox * c_p
            + oy * ow * c_p
            + ob * ow * oh * c_p
            + oz
        ] = color;
    }
}


__global__ void CONV_DW_OPT(const float* input, const half* kernel, const half* bias, float *output, const constBuffer* uConstant,
    DivModFast d_oc,
    DivModFast d_ow,
    DivModFast d_oh
    ) {
    float maxV = uConstant->maxValue;
    float minV = uConstant->minValue;
    int iw = uConstant->inputSize[0];
    int ih = uConstant->inputSize[1];
    int kw = uConstant->kernelSize[0];
    int kh = uConstant->kernelSize[1];
    int sw = uConstant->stride[0];
    int sh = uConstant->stride[1];
    int pw = uConstant->pad[0];
    int ph = uConstant->pad[1];
    int c = uConstant->channel;
    int c_p = c * PACK_NUMBER;

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < uConstant->total; index += blockDim.x * gridDim.x) {
        int oz, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color = bias[oz];
        int fxSta = max(0, -ix);
        int fySta = max(0, -iy);
        int fxEnd = min(kw, iw - ix);
        int fyEnd = min(kh, ih - iy);
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx + ix;
                float inp = input[0
                + sx * c_p
                + sy * iw * c_p
                + ob * iw * ih * c_p
                + oz
            ];
            float ker = kernel[0
                + fx
                + fy * kw
                + oz * kw * kh
            ];
            color = color + inp * ker;
        }
    }
    color = max(color, minV);
    color = min(color, maxV);

    output[index] = color;
}
}

static std::shared_ptr<ConvDepthWiseExecution::Resource> _makeResource(const Op* op, Backend* bn) {
    std::shared_ptr<ConvDepthWiseExecution::Resource> res(new ConvDepthWiseExecution::Resource);
    auto pool = static_cast<CUDABackend*>(bn)->getStaticBufferPool();
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();
    auto conv = op->main_as_Convolution2D();
    auto convCommon = conv->common();
    int kernelX = convCommon->kernelX();
    int kernelY = convCommon->kernelY();
    int depth = convCommon->outputCount();
    int depthC = UP_DIV(depth, PACK_NUMBER);
    res->weightTensor.reset(Tensor::createDevice<float>({kernelX * kernelY * depthC * PACK_NUMBER}));
    bool success = bn->onAcquireBuffer(res->weightTensor.get(), Backend::STATIC);
    if (!success) {
        return nullptr;
    }
    res->mFilter = (void *)res->weightTensor.get()->buffer().device;
    FuseRegion reg;
    int offset[8 * PACK_NUMBER];
    auto regionStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(sizeof(FuseRegion));
    auto offsetGpuStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(sizeof(offset));
    auto offsetGpu = (uint8_t*)offsetGpuStorage.first + offsetGpuStorage.second;
    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, conv, &filterDataPtr, &weightSize);
    auto tempWeightStorage = pool->alloc(weightSize * sizeof(float));
    auto tempWeight = (uint8_t*)tempWeightStorage.first + tempWeightStorage.second;
    cuda_check(cudaMemcpy(tempWeight, filterDataPtr, weightSize*sizeof(float), cudaMemcpyHostToDevice));
    reg.size[0] = 1;
    reg.size[1] = depthC * PACK_NUMBER;
    reg.size[2] = kernelY * kernelX;
    reg.srcStride[0] = 0;
    reg.srcStride[1] = kernelY * kernelX;
    reg.srcStride[2] = 1;
    reg.dstStride[0] = 0;
    reg.dstStride[1] = kernelY * kernelX;
    reg.dstStride[2] = 1;
    offset[0] = 1;
    offset[1] = depth;
    offset[2] = kernelY * kernelX;
    offset[3] = 0;
    offset[4] = 1;
    offset[5] = reg.size[1];
    offset[6] = reg.size[2];
    offset[7] = 0;
    reg.fuseNumber = 1;

    runtime->memcpy((uint8_t*)regionStorage.first + regionStorage.second, &reg, sizeof(FuseRegion), MNNMemcpyHostToDevice, true);
    runtime->memcpy(offsetGpu, offset, 8 * sizeof(int), MNNMemcpyHostToDevice, true);
    FuseRasterBlitFloatToHalf((uint8_t*)res->mFilter, (uint8_t*)tempWeight, (FuseRegion*)((uint8_t*)regionStorage.first + regionStorage.second), offsetGpu, runtime);
    pool->free(tempWeightStorage);
    res->biasTensor.reset(Tensor::createDevice<float>({depthC * PACK_NUMBER}));
    success = bn->onAcquireBuffer(res->biasTensor.get(), Backend::STATIC);
    res->mBias = (void *)res->biasTensor.get()->buffer().device;
    if (!success) {
        return nullptr;
    }
    if(conv->bias() != nullptr) {
        auto tempBiasStorage = pool->alloc(depth * sizeof(float));
        auto tempBias = (uint8_t*)tempBiasStorage.first + tempBiasStorage.second;
        cuda_check(cudaMemcpy(tempBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
        reg.size[0] = 1;
        reg.size[1] = 1;
        reg.size[2] = depthC * PACK_NUMBER;
        reg.srcStride[0] = 0;
        reg.srcStride[1] = 0;
        reg.srcStride[2] = 1;
        reg.dstStride[0] = 0;
        reg.dstStride[1] = 0;
        reg.dstStride[2] = 1;
        offset[0] = 1;
        offset[1] = 1;
        offset[2] = conv->bias()->size();
        offset[3] = 0;
        offset[4] = 1;
        offset[5] = 1;
        offset[6] = reg.size[2];
        offset[7] = 0;
        reg.fuseNumber = 1;
        runtime->memcpy((uint8_t*)regionStorage.first + regionStorage.second, &reg, sizeof(FuseRegion), MNNMemcpyHostToDevice, true);
        runtime->memcpy(offsetGpu, offset, 8 * sizeof(int), MNNMemcpyHostToDevice, true);
        FuseRasterBlitFloatToHalf((uint8_t*)res->mBias, (uint8_t*)tempBias, (FuseRegion*)((uint8_t*)regionStorage.first + regionStorage.second), offsetGpu, runtime);
        pool->free(tempBiasStorage);
    }
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(regionStorage);
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(offsetGpuStorage);
    return res;
}

ConvDepthWiseExecution::ConvDepthWiseExecution(const Op* op, Backend* bn, std::shared_ptr<Resource> resource) : Execution(bn) {
    mOp = op;
    mResource = resource;
    auto pool = static_cast<CUDABackend*>(bn)->getStaticBufferPool();
    mConstBuffer = pool->alloc(sizeof(constBuffer));
}
ConvDepthWiseExecution::~ ConvDepthWiseExecution() {
    auto pool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
    pool->free(mConstBuffer);
}

ErrorCode ConvDepthWiseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto pad = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mOp->main_as_Convolution2D()->common());
    auto conv = mOp->main_as_Convolution2D();
    auto convCommon = mOp->main_as_Convolution2D()->common();
    int channel = inputs[0]->channel();
    int channelDiv = UP_DIV(channel, PACK_NUMBER);
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
    parameters.channel = channelDiv;
    parameters.outputSize[0] = outputs[0]->width();
    parameters.outputSize[1] = outputs[0]->height();
    parameters.batch = inputs[0]->batch();

    parameters.total = parameters.batch * parameters.outputSize[1] * parameters.outputSize[0] * parameters.channel * PACK_NUMBER;
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        // Do nothing
    } else {
        parameters.minValue = -FLT_MAX;
        parameters.maxValue = FLT_MAX;
    }
    if (convCommon->relu()) {
        parameters.minValue = 0.0f;
    }
    if (convCommon->relu6()) {
        parameters.minValue = 0.0f;
        parameters.maxValue = 6.0f;
    }

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    runtime->memcpy((uint8_t*)mConstBuffer.first + mConstBuffer.second, &parameters, sizeof(constBuffer), MNNMemcpyHostToDevice);
    mTotalCount = parameters.total;
    //printf("%d-%d-%d-%d, %d-%d-%d-%d-%d\n", parameters.kernelSize[0], parameters.kernelSize[1], parameters.stride[0], parameters.stride[1], parameters.inputSize[0], parameters.inputSize[1], channel, parameters.outputSize[0], parameters.outputSize[1]);
    return NO_ERROR;
}

ErrorCode ConvDepthWiseExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto& prop = runtime->prop();
    int limitThreads = UP_DIV(mTotalCount, prop.multiProcessorCount);
    int threads_num = ALIMIN(prop.maxThreadsPerBlock/2, limitThreads);
    int block_num = prop.multiProcessorCount;
    auto constPtr = (uint8_t*)mConstBuffer.first + mConstBuffer.second;
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        if (inputs.size() == 1) {
            CONV_DW<<<block_num, threads_num>>>((const half*)inputs[0]->deviceId(), (const half*)mResource->mFilter,
                (const half*)mResource->mBias, (half*)outputs[0]->deviceId(), (const constBuffer*)(constPtr));
        }
        return NO_ERROR;
    }

    if (inputs.size() == 1) {
        // block_num = runtime->blocks_num(mTotalCount);
        // threads_num = runtime->threads_num();
        if(parameters.dilate[0] == 1 && parameters.dilate[1] == 1) {
            const int area = parameters.outputSize[0] * parameters.outputSize[1];
            DivModFast d_oc(parameters.channel * PACK_NUMBER);
            DivModFast d_ow(parameters.outputSize[0]);
            DivModFast d_oh(parameters.outputSize[1]);
            
            CONV_DW_OPT<<<block_num, threads_num>>>((const float*)inputs[0]->deviceId(), (const half*)mResource->mFilter,
                (const half*)mResource->mBias, (float*)outputs[0]->deviceId(), (const constBuffer*)(constPtr),
                d_oc, d_ow, d_oh);
        } else {
            CONV_DW<<<block_num, threads_num>>>((const float*)inputs[0]->deviceId(), (const half*)mResource->mFilter,
                (const half*)mResource->mBias, (float*)outputs[0]->deviceId(), (const constBuffer*)(constPtr));
        }
    }
    return NO_ERROR;
}

class ConvDepthWiseExecutionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (inputs.size() > 1) {
            return nullptr;
        }
        auto res = _makeResource(op, backend);
        if (nullptr == res) {
            return nullptr;
        }
        return new ConvDepthWiseExecution(op, backend, res);
    }
};

static CUDACreatorRegister<ConvDepthWiseExecutionCreator> __init(OpType_ConvolutionDepthwise);
}
}