#include "InterpExecution.hpp"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void INTERP(const int n, const int ih, const int iw, const int oh, const int ow, 
    const float scaleh, const float scalew, const float offseth, const float offsetw, const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % ow;
        int tmp = index / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = min(max(0, (int)floor((float)x*scalew+offsetw)), iw-1);
        int iy = min(max(0, (int)floor((float)y*scaleh+offseth)), ih-1);
        out[z*oh*ow + y*oh + x] = in[z*ih*iw + iy*ih + ix];
    }
}

InterpExecution::InterpExecution(const Interp* interp, Backend *backend) : Execution(backend) {
    mWidthOffset  = interp->widthOffset();
    mHeightOffset = interp->heightOffset();
    mResizeType   = interp->resizeType();
    mScaleWidth   = interp->widthScale();
    mScaleHeight  = interp->heightScale();
}
InterpExecution::~InterpExecution() {
    //do nothing
}

ErrorCode InterpExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    //MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];

    mChannel = input->channel();
    mBatch   = input->batch();

    mInputHeight  = input->height();
    mInputWidth   = input->width();
    mOutputHeight = output->height();
    mOutputWidth  = output->width();

    mCount = mBatch*mChannel*mOutputHeight*mOutputWidth;
    //printf("%d mInputHeight:%d- mInputWidth:%d- mOutputHeight:%d- mOutputWidth:%d, mScaleHeight:%f- mScaleWidth:%f\n", inputs.size(), mInputHeight,mInputWidth,mOutputHeight, mOutputWidth, mScaleHeight, mScaleWidth);
    return NO_ERROR;
}

ErrorCode InterpExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();

    INTERP<<<block_num, threads_num>>>(mCount, mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,
        mScaleHeight, mScaleWidth, mHeightOffset, mWidthOffset, (const float *)input_addr, (float *)output_addr);
    return NO_ERROR;
}

class InterpCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_Interp();
        if(param->resizeType() != 1) {
            MNN_PRINT("CUDA interp resize type:%d not support, back to CPU\n", param->resizeType());
            return nullptr;
        }
        return new InterpExecution(param, backend);
    }
};

static CUDACreatorRegister<InterpCreator> __init(OpType_Interp);

}
}