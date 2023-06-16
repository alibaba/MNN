//
//  Int8ToFloatExecution.cu
//  MNN
//
//  Created by MNN on 2023/01/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef ENABLE_CUDA_QUANT

#include "Int8ToFloatExecution.hpp"
#include "../MNNCUDADefine.hpp"
#include "../MNNCUDAFunction.cuh"
namespace MNN {
namespace CUDA {

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
__global__ void INT8_2_FLOAT(const int total,
    const int channelsPackInt8,
    const int channelsPackFloat,
    const int channels, 
    const int8_t* in, 
    T* out,
    const float* scaleData, 
    const int8_t zeroPoint,
    DivModFast d_cp
) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);

        int idx_inp = nhw_idx * channelsPackInt8 + 4*c_idx;
        char4 inp_0 = ((char4 *)(in + idx_inp))[0];
        float4 scale_0 = ((float4 *)(scaleData + (c_idx << 2)))[0];

        const int idx_out = index << 2;
        
        out[idx_out+0] = (T)((inp_0.x - zeroPoint) * scale_0.x);
        out[idx_out+1] = (T)((inp_0.y - zeroPoint) * scale_0.y);
        out[idx_out+2] = (T)((inp_0.z - zeroPoint) * scale_0.z);
        out[idx_out+3] = (T)((inp_0.w - zeroPoint) * scale_0.w);
    }
}

template<typename T>
__global__ void INT8_2_FLOAT_SINGLE(const int total,
    const int channelsPackInt8,
    const int channelsPackFloat,
    const int channels, 
    const int8_t* in, 
    T* out,
    const float scaleData, 
    const int8_t zeroPoint,
    DivModFast d_cp
) {
    CUDA_KERNEL_LOOP(index, total) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);

        int idx_inp = nhw_idx * channelsPackInt8 + 4*c_idx;
        char4 inp_0 = ((char4 *)(in + idx_inp))[0];

        const int idx_out = index << 2;
        out[idx_out+0] = (T)((inp_0.x - zeroPoint) * scaleData);
        out[idx_out+1] = (T)((inp_0.y - zeroPoint) * scaleData);
        out[idx_out+2] = (T)((inp_0.z - zeroPoint) * scaleData);
        out[idx_out+3] = (T)((inp_0.w - zeroPoint) * scaleData);
    }
}

Int8ToFloatExecution::Int8ToFloatExecution(Backend *backend, const std::vector<Tensor *> &inputs, const MNN::Op *param) : Execution(backend) {
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();
    auto scale         = param->main_as_QuantizedFloatParam();

    const int scaleLen = scale->tensorScale()->size();
    mClipBits = scale->nbits();

    if (1 == scaleLen) {
        mSingle = true;
        mSingleScale = scale->tensorScale()->data()[0];
    } else {

        auto staticPool = static_cast<CUDABackend*>(backend)->getStaticBufferPool();
        mScaleStorage = staticPool->alloc(UP_DIV(scaleLen, PACK_NUMBER) * PACK_NUMBER * sizeof(float));
        mScales = (void*)((uint8_t*)mScaleStorage.first + mScaleStorage.second);
        runtime->memset(mScales, 0, UP_DIV(scaleLen, PACK_NUMBER) * PACK_NUMBER * sizeof(float));

        runtime->memcpy(mScales, scale->tensorScale()->data(), scaleLen * sizeof(float), MNNMemcpyHostToDevice);
    }

    mZeroPoint = scale->zeroPoint();
}
Int8ToFloatExecution::~Int8ToFloatExecution() {
    if(!mSingle) {
        auto staticPool = static_cast<CUDABackend*>(backend())->getStaticBufferPool();
        staticPool->free(mScaleStorage);
    }
}

ErrorCode Int8ToFloatExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
        MNN_ERROR("Int8ToFloatExecution not support format:%d\n", format);
        MNN_ASSERT(false);
    }

    mCount = mArea * UP_DIV(mChannel, PACK_NUMBER) * 2;
    // printf("Int8_2_Float size:%d-%d-%d\n\n", mArea, mChannel, mCount);
    return NO_ERROR;
}

ErrorCode Int8ToFloatExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
 
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();

    auto channelPackInt8 = UP_DIV(mChannel, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
    auto channelPackFloat = UP_DIV(mChannel, PACK_NUMBER) * 2;
    DivModFast cpD(channelPackFloat);

    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        if(mSingle) {
            INT8_2_FLOAT_SINGLE<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const int8_t *)input_addr, (half *)output_addr,\
                mSingleScale, mZeroPoint, cpD);
            checkKernelErrors;  
        } else {
            INT8_2_FLOAT<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const int8_t *)input_addr, (half *)output_addr,\
                (const float *)mScales, mZeroPoint, cpD);
            checkKernelErrors;  
        }
    } else {
        if(mSingle) {
            INT8_2_FLOAT_SINGLE<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const int8_t *)input_addr, (float *)output_addr,\
                mSingleScale, mZeroPoint, cpD);
            checkKernelErrors;
        } else {
            INT8_2_FLOAT<<<block_num, threads_num>>>(mCount, channelPackInt8, channelPackFloat, mChannel, (const int8_t *)input_addr, (float *)output_addr,\
                (const float *)mScales, mZeroPoint, cpD);
            checkKernelErrors;
        }
    }

    return NO_ERROR;
}

class Int8ToFloatCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if(op->main_as_QuantizedFloatParam() == nullptr) {
            return new CastWrapExecution(backend, DataType_DT_FLOAT);
        }
        return new Int8ToFloatExecution(backend, inputs, op);
    }
};

static CUDACreatorRegister<Int8ToFloatCreator> __init(OpType_Int8ToFloat);

}
}
#endif