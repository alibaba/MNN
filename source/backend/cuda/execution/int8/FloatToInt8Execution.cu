//
//  FloatToInt8Execution.cu
//  MNN
//
//  Created by MNN on 2023/01/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_QUANT

#include "FloatToInt8Execution.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void FLOAT_2_INT8(const int total,
    const int channelsPackInt8,
    const int channelsPackFloat, 
    const int channels, 
    const T* in, 
    int8_t* out,
    const float* scaleData, 
    const int8_t zeroPoint, 
    const int8_t clampMax, 
    const int8_t clampMin,
    DivModFast d_cp
) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);

        int out_idx = index << 2;
        if(4 * c_idx >= channels) {
            ((char4 *)(out + out_idx))[0] = make_char4(0, 0, 0, 0);
            continue;
        }

        float4 scale_0 = ((float4 *)(scaleData + (c_idx << 2)))[0];

        int idx_inp = nhw_idx * channelsPackFloat + 4*c_idx;

        float inp_0 = in[idx_inp];
        float inp_1 = in[idx_inp+1];
        float inp_2 = in[idx_inp+2];
        float inp_3 = in[idx_inp+3];

        int res = __float2int_rn(inp_0 * scale_0.x) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx] = res;

        res = __float2int_rn(inp_1 * scale_0.y) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 1] = res;

        res = __float2int_rn(inp_2 * scale_0.z) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 2] = res;


        res = __float2int_rn(inp_3 * scale_0.w) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 3] = res;
    }
}

template<typename T>
__global__ void FLOAT_2_INT8_SINGLE(const int total,
    const int channelsPackInt8,
    const int channelsPackFloat, 
    const int channels, 
    const T* in, 
    int8_t* out,
    const float scaleData, 
    const int8_t zeroPoint, 
    const int8_t clampMax, 
    const int8_t clampMin,
    DivModFast d_cp
) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);

        int out_idx = index << 2;
        if(4 * c_idx >= channels) {
            ((char4 *)(out + out_idx))[0] = make_char4(0, 0, 0, 0);
            continue;
        }

        int idx_inp = nhw_idx * channelsPackFloat + 4*c_idx;

        float inp_0 = in[idx_inp];
        float inp_1 = in[idx_inp+1];
        float inp_2 = in[idx_inp+2];
        float inp_3 = in[idx_inp+3];

        int res = __float2int_rn(inp_0 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx] = res;

        res = __float2int_rn(inp_1 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 1] = res;

        res = __float2int_rn(inp_2 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 2] = res;


        res = __float2int_rn(inp_3 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[out_idx + 3] = res;
    }
}

FloatToInt8Execution::FloatToInt8Execution(Backend *backend, const std::vector<Tensor *> &inputs, const MNN::Op *param) : Execution(backend) {
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    auto scale         = param->main_as_QuantizedFloatParam();

    const int scaleLen = scale->tensorScale()->size();
    mClipBits = scale->nbits();

    if (1 == scaleLen) {
        mSingle = true;
        mSingleScale = scale->tensorScale()->data()[0];
    } else {
        auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
        mScaleStorage = staticPool->alloc(UP_DIV(scaleLen, INT8_PACK_NUMBER) * INT8_PACK_NUMBER * sizeof(float));
        mScales = (void *)((uint8_t*)mScaleStorage.first + mScaleStorage.second);
        runtime->memset(mScales, 0, UP_DIV(scaleLen, INT8_PACK_NUMBER) * INT8_PACK_NUMBER * sizeof(float));

        runtime->memcpy(mScales, scale->tensorScale()->data(), scaleLen * sizeof(float), MNNMemcpyHostToDevice);
    }

    mZeroPoint = scale->zeroPoint();
    mClampMin = scale->clampMin();
    mClampMax = scale->clampMax();
}
FloatToInt8Execution::~FloatToInt8Execution() {
    if(!mSingle) {
        auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
        staticPool->free(mScaleStorage);
    }
}

ErrorCode FloatToInt8Execution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto dims = input->dimensions();
    MNN_ASSERT(dims >= 2);

    auto format = TensorUtils::getDescribe(input)->dimensionFormat;
    if (format == MNN_DATA_FORMAT_NHWC) {
        mChannel = input->length(dims-1);
        mArea = 1;
        for(int i = 0; i < dims-1; i++) {
            mArea *= input->length(i);
        }
    } else if(format == MNN_DATA_FORMAT_NCHW || format == MNN_DATA_FORMAT_NC4HW4) {
        mChannel = input->length(1);
        mArea = input->length(0);
        for(int i = 2; i < dims; i++) {
            mArea *= input->length(i);
        }
    } else {
        MNN_ERROR("FloatToInt8Execution not support format:%d\n", format);
        MNN_ASSERT(false);
    }
    
    mCount = mArea * UP_DIV(mChannel, INT8_PACK_NUMBER) * 4;
    // printf("mChannel:%d- mArea:%d- mCount:%d, format:%d\n",mChannel,mArea, mCount, format);
    return NO_ERROR;
}

ErrorCode FloatToInt8Execution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();
    
    auto channelPackInt8 = UP_DIV(mChannel, INT8_PACK_NUMBER) * 4;
    auto channelPackFloat = UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER;
    DivModFast cpD(channelPackInt8);

    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        if(mSingle) {
            FLOAT_2_INT8_SINGLE<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const half *)input_addr, (int8_t *)output_addr,\
                mSingleScale, mZeroPoint, mClampMax, mClampMin, cpD);
            checkKernelErrors;
        } else {
            FLOAT_2_INT8<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const half *)input_addr, (int8_t *)output_addr,\
                (const float *)mScales, mZeroPoint, mClampMax, mClampMin, cpD);
            checkKernelErrors;
        }
    } else {
        if(mSingle) {
            FLOAT_2_INT8_SINGLE<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const float *)input_addr, (int8_t *)output_addr,\
                mSingleScale, mZeroPoint, mClampMax, mClampMin, cpD);
            checkKernelErrors;
        } else {
            FLOAT_2_INT8<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const float *)input_addr, (int8_t *)output_addr,\
                (const float *)mScales, mZeroPoint, mClampMax, mClampMin, cpD);
            checkKernelErrors;
        }
    }

    return NO_ERROR;
}

class FloatToInt8Creator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if(op->main_as_QuantizedFloatParam() == nullptr) {
            return new CastWrapExecution(backend, DataType_DT_INT8);
        }
        return new FloatToInt8Execution(backend, inputs, op);
    }
};

static CUDACreatorRegister<FloatToInt8Creator> __init(OpType_FloatToInt8);

}
}

#endif