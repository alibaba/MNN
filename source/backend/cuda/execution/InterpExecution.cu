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
        out[z*oh*ow + y*ow + x] = in[z*ih*iw + iy*iw + ix];
    }
}

template<typename T>
__global__ void INTERP_BILINEAR(const int n, const int ih, const int iw, const int oh, const int ow, 
    const float scaleh, const float scalew, const float offseth, const float offsetw, const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, n) {
        int x = index % ow;
        int tmp = index / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        float fx = x*scalew+offsetw;
        int ix_0 = min(max(0, (int)floor(fx)), iw-1);
        int ix_1 = min((int)ceil(fx), iw-1);
        float fy = y*scaleh+offseth;
        int iy_0 = min(max(0, (int)floor(fy)), ih-1);
        int iy_1 = min((int)ceil(fy), ih-1);

        int index_00 = z*ih*iw + iy_0*iw + ix_0;
        int index_01 = z*ih*iw + iy_0*iw + ix_1;
        int index_10 = z*ih*iw + iy_1*iw + ix_0;
        int index_11 = z*ih*iw + iy_1*iw + ix_1;

        float factor_x = fx-ix_0;
        float factor_y = fy-iy_0;
        out[z*oh*ow + y*ow + x] = (1.0-factor_x)*(1.0-factor_y)*in[index_00] + factor_x*(1.0-factor_y)*in[index_01] +
                                  (1.0-factor_x)*factor_y*in[index_10] + factor_x*factor_y*in[index_11];
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
    //printf("mBatch:%d-mChannel:%d-mInputHeight:%d- mInputWidth:%d- mOutputHeight:%d- mOutputWidth:%d, mScaleHeight:%f- mScaleWidth:%f %f %f\n", mBatch, mChannel, mInputHeight,mInputWidth,mOutputHeight, mOutputWidth, mScaleHeight, mScaleWidth, mWidthOffset, mHeightOffset);
    return NO_ERROR;
}

ErrorCode InterpExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();

    if(mResizeType == 1){
        INTERP<<<block_num, threads_num>>>(mCount, mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,
            mScaleHeight, mScaleWidth, mHeightOffset, mWidthOffset, (const float *)input_addr, (float *)output_addr);
    } else if(mResizeType == 2) {
        INTERP_BILINEAR<<<block_num, threads_num>>>(mCount, mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,
            mScaleHeight, mScaleWidth, mHeightOffset, mWidthOffset, (const float *)input_addr, (float *)output_addr);       
    }
    return NO_ERROR;
}

class InterpCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_Interp();
        if(param->resizeType() != 1 && param->resizeType() != 2) {
            MNN_PRINT("CUDA interp resize type:%d not support, back to CPU\n", param->resizeType());
            return nullptr;
        }
        return new InterpExecution(param, backend);
    }
};

static CUDACreatorRegister<InterpCreator> __init(OpType_Interp);

}
}
