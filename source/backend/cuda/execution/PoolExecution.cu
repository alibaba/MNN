#include "PoolExecution.hpp"
namespace MNN {
namespace CUDA {
template <typename T>
__global__ void avgpool(const T* uInput, T* uOutput,
        int bc,
        int ih, int iw,
        int oh, int ow,
        int padX, int padY,
        int kernelX, int kernelY,
        int strideX, int strideY
        ) {
    int total = bc * oh * ow;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        T sumValue = (T)0;
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx)
            {
                int currentX = ix + fx;
                int currentY = iy + fy;
                T inputColor = uInput[0
                + z * iw * ih
                + currentY * iw
                + currentX
                ];
                sumValue = sumValue + inputColor;
            }
        }
        uOutput[0
            + z * ow * oh
            + y * ow
            + x
        ] = sumValue / ((T)(ey-sy)*(T)(ex-sx));
    }
}
template <typename T>
__global__ void maxpool(const T* uInput, T* uOutput,
        int bc,
        int ih, int iw,
        int oh, int ow,
        int padX, int padY,
        int kernelX, int kernelY,
        int strideX, int strideY
        ) {
    int total = bc * oh * ow;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        int x = i % ow;
        int tmp = i / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = x * strideX - padX;
        int iy = y * strideY - padY;
        int sx = max(0, -ix);
        int sy = max(0, -iy);
        int ex = min(kernelX, iw - ix);
        int ey = min(kernelY, ih - iy);
        T maxValue = (T)(-1000000);
        for (int fy=sy; fy<ey; ++fy) {
            for (int fx=sx; fx<ex; ++fx)
            {
                int currentX = ix + fx;
                int currentY = iy + fy;
                T inputColor = uInput[0
                + z * iw * ih
                + currentY * iw
                + currentX
                ];
                maxValue = max(inputColor, maxValue);
            }
        }
        uOutput[0
            + z * ow * oh
            + y * ow
            + x
        ] = maxValue;
    }
}
ErrorCode PoolExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto layer       = mParameter;
    int strideWidth  = layer->strideX();
    int strideHeight = layer->strideY();
    int padWidth     = layer->padX();
    int padHeight    = layer->padY();

    // edit const if global
    auto input       = inputs[0];
    auto output      = outputs[0];
    int kernelWidth  = std::min(layer->kernelX(), input->width());
    int kernelHeight = std::min(layer->kernelY(), input->height());
    if (layer->isGlobal()) {
        kernelWidth  = input->width();
        kernelHeight = input->height();
        strideWidth  = input->width();
        strideHeight = input->height();
        padWidth     = 0;
        padHeight    = 0;
    }
    if (layer->padType() == PoolPadType_SAME) {
        int padNeededWidth  = (output->width() - 1) * strideWidth + kernelWidth - input->width();
        int padNeededHeight = (output->height() - 1) * strideHeight + kernelHeight - input->height();
        padWidth            = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
        padHeight           = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
    } else if (layer->padType() == PoolPadType_VALID) {
        padWidth = padHeight = 0;
    }
    mPoolType      = layer->type();
    auto padType           = layer->padType();
    if (layer->pads() != nullptr && padType == PoolPadType_CAFFE) {
        padType = PoolPadType_VALID;
    }
    mPadType = padType;
    mPaddings = {padWidth, padHeight};
    mStrides = {strideWidth, strideHeight};
    mKernels = {kernelWidth, kernelHeight};
    return NO_ERROR;
}

ErrorCode PoolExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto iw = inputs[0]->width();
    auto ih = inputs[0]->height();
    auto bc = inputs[0]->batch() * inputs[0]->channel();
    auto ow = outputs[0]->width();
    auto oh = outputs[0]->height();
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    int block_num = runtime->blocks_num(bc * ow * oh);
    int threads_num = runtime->threads_num();
    auto inputPtr = (const float*)inputs[0]->deviceId();
    auto outputPtr = (float*)outputs[0]->deviceId();
    switch (mPoolType) {
        case PoolType_AVEPOOL:
            avgpool<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                bc, 
                ih, iw,
                oh, ow,
                mPaddings[0], mPaddings[1],
                mKernels[0], mKernels[1],
                mStrides[0], mStrides[1]
                );
            return NO_ERROR;
        case PoolType_MAXPOOL:
            maxpool<<<block_num, threads_num>>>(inputPtr, outputPtr, 
                bc, 
                ih, iw,
                oh, ow,
                mPaddings[0], mPaddings[1],
                mKernels[0], mKernels[1],
                mStrides[0], mStrides[1]
                );
            return NO_ERROR;
    }
    return NOT_SUPPORT;
}
class PoolCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new PoolExecution(op->main_as_Pool(), backend);
    }
};

static CUDACreatorRegister<PoolCreator> __init(OpType_Pooling);


};
};