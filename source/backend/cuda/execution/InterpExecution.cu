#include "InterpExecution.hpp"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {
#define CUDA_KERNEL_LOOP(i, n) for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void INTERP_NERAEST(const int total, const int c_p, 
    const int ih, const int iw, const int oh, const int ow, 
    const float scaleh, const float scalew, const float offseth, const float offsetw, 
    const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, total) {
        int tmp0 = index / c_p;
        int c_idx = index % c_p;
        int x = tmp0 % ow;
        int tmp = tmp0 / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = min(max(0, (int)floor((float)x * scalew + offsetw)), iw-1);
        int iy = min(max(0, (int)floor((float)y * scaleh + offseth)), ih-1);
        out[((z * oh + y) * ow + x) * c_p + c_idx]
            = in[((z * ih + iy) * iw + ix) * c_p + c_idx];
    }
}

template<typename T>
__global__ void INTERP_NERAEST_ROUND(const int total, const int c_p, 
    const int ih, const int iw, const int oh, const int ow, 
    const float scaleh, const float scalew, const float offseth, const float offsetw, 
    const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, total) {
        int tmp0 = index / c_p;
        int c_idx = index % c_p;
        int x = tmp0 % ow;
        int tmp = tmp0 / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        int ix = min(max(0, (int)floor((float)x * scalew + offsetw + 0.499f)), iw-1);
        int iy = min(max(0, (int)floor((float)y * scaleh + offseth + 0.499f)), ih-1);
        out[((z * oh + y) * ow + x) * c_p + c_idx]
            = in[((z * ih + iy) * iw + ix) * c_p + c_idx];
    }
}

template<typename T>
__global__ void INTERP_BILINEAR(const int total, const int c_p, 
    const int ih, const int iw, const int oh, const int ow, 
    const float scaleh, const float scalew, const float offseth, const float offsetw,
    const T* in, T* out) {
    CUDA_KERNEL_LOOP(index, total) {
        int tmp0 = index / c_p;
        int c_idx = index % c_p;
        int x = tmp0 % ow;
        int tmp = tmp0 / ow;
        int y = tmp % oh;
        int z = tmp / oh;
        float fx = x*scalew+offsetw;
        int ix_0 = min(max(0, (int)floor(fx)), iw-1);
        int ix_1 = min((int)ceil(fx), iw-1);
        float fy = y*scaleh+offseth;
        int iy_0 = min(max(0, (int)floor(fy)), ih-1);
        int iy_1 = min((int)ceil(fy), ih-1);

        int index_00 = (z * ih + iy_0) * iw + ix_0;
        int index_01 = (z * ih + iy_0) * iw + ix_1;
        int index_10 = (z * ih + iy_1) * iw + ix_0;
        int index_11 = (z * ih + iy_1) * iw + ix_1;
        index_00 = index_00 * c_p + c_idx;
        index_01 = index_01 * c_p + c_idx;
        index_10 = index_10 * c_p + c_idx;
        index_11 = index_11 * c_p + c_idx;

        float factor_x = fx-ix_0;
        float factor_y = fy-iy_0;
        out[((z * oh + y) * ow + x) * c_p + c_idx] =
            (1.0-factor_x)*(1.0-factor_y)*(float)in[index_00] 
            + factor_x*(1.0-factor_y)*(float)in[index_01] 
            + (1.0-factor_x)*factor_y*(float)in[index_10] 
            + factor_x*factor_y*(float)in[index_11];
    }
}

/* FIXME : TODO */
template<typename T>
__global__ void INTERP_BILINEAR_OPT(const int n, const int ih, const int iw, const int oh, const int ow, 
    const float scaleh, const float scalew, const float offseth, const float offsetw, const T* in, T* out,
    DivModFast d_ow, DivModFast d_oh) {
    CUDA_KERNEL_LOOP(total, n) {
        size_t index = total >> 4;
        size_t remain = total & 15;

        int tmp, x_idx, y, z;
        d_ow.divmod(index, tmp, x_idx);
        d_oh.divmod(tmp, z, y);

        size_t x = x_idx << 1;
        float fx = x*scalew+offsetw;
        int ix_0 = min(max(0, (int)floor(fx)), iw-1);
        int ix_1 = min((int)ceil(fx), iw-1);

        float fx_1 = fx + scalew;
        int ix_2 = min(max(0, (int)floor(fx_1)), iw-1);
        int ix_3 = min((int)ceil(fx_1), iw-1);

        float fy = y*scaleh+offseth;
        int iy_0 = min(max(0, (int)floor(fy)), ih-1);
        int iy_1 = min((int)ceil(fy), ih-1);

        int index_00 = (z*ih+ iy_0)*iw + ix_0;
        int index_01 = index_00 - ix_0 + ix_1;
        int index_10 = (z*ih+ iy_1)*iw + ix_0;
        int index_11 = index_10 - ix_0 + ix_1;
        index_00 = (index_00 << 4) + remain;
        index_01 = (index_01 << 4) + remain;
        index_10 = (index_10 << 4) + remain;
        index_11 = (index_11 << 4) + remain;

        float factor_x = fx-ix_0;
        float factor_y = fy-iy_0;
        float in_00 = (float)in[index_00];
        float in_01 = (float)in[index_01];
        float in_10 = (float)in[index_10];
        float in_11 = (float)in[index_11];

        float factor_00 = (1.0-factor_x)*(1.0-factor_y);
        float factor_01 = factor_x*(1.0-factor_y);
        float factor_10 = (1.0-factor_x)*factor_y;
        float factor_11 = factor_x*factor_y;

        size_t dstOffset = (((z*oh+ y)*ow + x) << 4) + remain;
        out[dstOffset] = \
            factor_00* in_00 + factor_01*in_01 + \
            factor_10* in_10 + factor_11*in_11;

        if(x+1 >= ow) {
            continue;
        }

        if(ix_2 != ix_0) {
            index_00 = index_00 + ((ix_2-ix_0) << 4);
            index_10 = index_10 + ((ix_2-ix_0) << 4);
            in_00 = (float)in[index_00];
            in_10 = (float)in[index_10];
        }
        if(ix_3 != ix_1) {
            index_01 = index_01 + ((ix_3-ix_1) << 4);
            index_11 = index_11 + ((ix_3-ix_1) << 4);
            in_01 = (float)in[index_01];
            in_11 = (float)in[index_11];
        }

        if(factor_x != fx_1-ix_2) {
            factor_x = fx_1-ix_2;
            factor_00 = (1.0-factor_x)*(1.0-factor_y);
            factor_01 = factor_x*(1.0-factor_y);
            factor_10 = (1.0-factor_x)*factor_y;
            factor_11 = factor_x*factor_y;
        }
        out[dstOffset+ PACK_NUMBER] = \
            factor_00* in_00 + factor_01*in_01 + \
            factor_10* in_10 + factor_11*in_11;
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

    mCount = mBatch*UP_DIV(mChannel, PACK_NUMBER)*mOutputHeight*mOutputWidth * PACK_NUMBER;
    //MNN_PRINT("mBatch:%d-mChannel:%d-mInputHeight:%d- mInputWidth:%d- mOutputHeight:%d- mOutputWidth:%d, mScaleHeight:%f- mScaleWidth:%f %f %f\n", mBatch, mChannel, mInputHeight,mInputWidth,mOutputHeight, mOutputWidth, mScaleHeight, mScaleWidth, mWidthOffset, mHeightOffset);
    return NO_ERROR;
}

ErrorCode InterpExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();
    //MNN_PRINT("Interp type:%d\n", mResizeType);
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        if(mResizeType == 1){
            INTERP_NERAEST<<<block_num, threads_num>>>(mCount, UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER,
                mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,
                mScaleHeight, mScaleWidth, mHeightOffset, mWidthOffset, (const half *)input_addr, (half *)output_addr);
        } else if(mResizeType == 2) {
            INTERP_BILINEAR<<<block_num, threads_num>>>(mCount, UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER, 
                mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,\
                mScaleHeight, mScaleWidth, mHeightOffset, mWidthOffset, (const half *)input_addr, (half *)output_addr);

            if(0) { // TO USE after fixed
                mCount = mBatch*UP_DIV(mChannel, PACK_NUMBER)*mOutputHeight*((mOutputWidth+1)/ 2) * PACK_NUMBER;
                block_num = runtime->blocks_num(mCount);
                threads_num = runtime->threads_num();

                DivModFast d_ow((mOutputWidth+1)/2);
                DivModFast d_oh(mOutputHeight);
                INTERP_BILINEAR_OPT<<<block_num, threads_num>>>(mCount, mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,\
                    mScaleHeight, mScaleWidth, mHeightOffset, mWidthOffset, (const half *)input_addr, (half *)output_addr, d_ow, d_oh);
            }   

        } else if (mResizeType == 4) {
            INTERP_NERAEST_ROUND<<<block_num, threads_num>>>(mCount, UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER,
                mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,
                mScaleHeight, mScaleWidth, mHeightOffset, mWidthOffset, (const half *)input_addr, (half *)output_addr);
        }
        return NO_ERROR;
    }

    if(mResizeType == 1){
        INTERP_NERAEST<<<block_num, threads_num>>>(mCount, UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER,
            mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,
            mScaleHeight, mScaleWidth, mHeightOffset, mWidthOffset, (const float *)input_addr, (float *)output_addr);
    } else if(mResizeType == 2) {
        INTERP_BILINEAR<<<block_num, threads_num>>>(mCount, UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER, 
            mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,
            mScaleHeight, mScaleWidth, mHeightOffset, mWidthOffset, (const float *)input_addr, (float *)output_addr);       
    } else if (mResizeType == 4) {
        INTERP_NERAEST_ROUND<<<block_num, threads_num>>>(mCount, UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER,
            mInputHeight, mInputWidth, mOutputHeight, mOutputWidth,
            mScaleHeight, mScaleWidth, mHeightOffset, mWidthOffset, (const float *)input_addr, (float *)output_addr);
    }
    return NO_ERROR;
}

class InterpCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_Interp();
        if(param->resizeType() == 3) {
            MNN_PRINT("CUDA interp resize type:%d not support, back to CPU\n", param->resizeType());
            return nullptr;
        }
        return new InterpExecution(param, backend);
    }
};

static CUDACreatorRegister<InterpCreator> __init(OpType_Interp);

}
}
