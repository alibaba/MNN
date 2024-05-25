//
//  CastExecution.cpp
//  MNN
//
//  Created by MNN on 2023/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CastExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "Raster.cuh"
#include "backend/cuda/core/CUDABackend.hpp"
#include "MNNCUDAFunction.cuh"
#include "MNNCUDADefine.hpp"

namespace MNN {
namespace CUDA {

template <typename T1, typename T2>
__global__ void CAST(T1 *input, T2 *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = (T2)(input[i]);
  }
  return;
}

template <typename T1, typename T2>
__global__ void CASTMIDFLOAT(T1 *input, T2 *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = (T2)((float)input[i]);
  }
  return;
}

template <typename T>
__global__ void BF162FLOAT(int16_t *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    float tmp;
    ((int16_t *)&tmp)[0] = 0;
    ((int16_t *)&tmp)[1] = input[i];
    output[i] = (T)tmp;
  }
}

__global__ void CASTBOOL(int32_t *input, int32_t *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i] > 0 ? 1 : 0;
  }
  return;
}

template<typename T>
__global__ void FLOAT_2_INT8_CAST(const int count,
    const T* in, 
    int8_t* out,
    const float scaleData, 
    const int8_t zeroPoint, 
    const int8_t clampMax, 
    const int8_t clampMin
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (count); index += blockDim.x * gridDim.x) {
        float inp_0 = in[index];
        int res = __float2int_rn(inp_0 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[index] = res;
    }
}

template<typename T>
__global__ void INT8_2_FLOAT_CAST(const int count,
    const int8_t* in, 
    T* out,
    const float scaleData, 
    const int8_t zeroPoint
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (count); index += blockDim.x * gridDim.x) {
        char inp_0 = in[index];
        out[index] = (T)((inp_0 - zeroPoint) * scaleData);
    }
}

template<typename T>
__global__ void FLOAT_2_INT8_CAST_PACK(const int count,
    const T* in, 
    int8_t* out,
    const float scaleData, 
    const int8_t zeroPoint, 
    const int8_t clampMax, 
    const int8_t clampMin,
    const int channelPackFloat,
    const int channels,
    DivModFast d_cp
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (count); index += blockDim.x * gridDim.x) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);
        if(c_idx >= channels) {
            out[index] = 0;
            return;
        }
        float inp_0 = in[nhw_idx * channelPackFloat + c_idx];
        int res = __float2int_rn(inp_0 * scaleData) + zeroPoint;
        res = min(res, clampMax);
        res = max(res, clampMin);

        out[index] = res;
    }
}

template<typename T>
__global__ void INT8_2_FLOAT_CAST_PACK(const int count,
    const int8_t* in, 
    T* out,
    const float scaleData, 
    const int8_t zeroPoint,
    const int channelPackInt8,
    const int channels,
    DivModFast d_cp
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (count); index += blockDim.x * gridDim.x) {
        int nhw_idx, c_idx;
        d_cp.divmod(index, nhw_idx, c_idx);

        char inp_0 = in[nhw_idx * channelPackInt8 + c_idx];
        out[index] = (T)((inp_0 - zeroPoint) * scaleData);
    }
}

static DataType _mapDataType(DataType src) {
    if (DataType_DT_BOOL == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_INT64 == src) {
        return DataType_DT_INT32;
    }
    if (DataType_DT_DOUBLE == src) {
        return DataType_DT_FLOAT;
    }
    return src;
}

ErrorCode CastExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto count = CUDABackend::realSize(inputs[0]);
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    auto input = inputs[0]->deviceId();
    auto output = outputs[0]->deviceId();
    auto dstT = _mapDataType(mDst);

    const auto &inputDataType = inputs[0]->getType();
    if (inputDataType.bytes() == 4 && mDst == MNN::DataType_DT_BOOL) {
        CASTBOOL<<<block_num, threads_num>>>((int32_t*)input, (int32_t*)output, count);
        checkKernelErrors;
        return NO_ERROR;
    }
    if (inputs[0]->buffer().type == outputs[0]->buffer().type) {
        runtime->memcpy((void*)output, (void*)input, count * static_cast<CUDABackend*>(backend())->getBytes(inputs[0]), MNNMemcpyDeviceToDevice, true);
        checkKernelErrors;
        return NO_ERROR;
    }
    if (dstT == MNN::DataType_DT_INT32 && halide_type_of<int8_t>() == inputDataType) {
        CAST<<<block_num, threads_num>>>((int8_t*)input, (int32_t*)output, count);
        checkKernelErrors;
        return NO_ERROR;
    } else if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<int32_t>() == inputDataType) {
        CAST<<<block_num, threads_num>>>((int32_t*)input, (uint8_t*)output, count);
        checkKernelErrors;
        return NO_ERROR;
    } else if (dstT == MNN::DataType_DT_INT32 && halide_type_of<uint8_t>() == inputDataType) {
        CAST<<<block_num, threads_num>>>((uint8_t*)input, (int32_t*)output, count);
        checkKernelErrors;
        return NO_ERROR;
    }
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        if (dstT == MNN::DataType_DT_INT32 && halide_type_of<float>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((half*)input, (int*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int32_t>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((int*)input, (half*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<uint8_t>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((uint8_t*)input, (half*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int8_t>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((int8_t*)input, (half*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_INT8 && halide_type_of<float>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((half*)input, (int8_t*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<float>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((half*)input, (uint8_t*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_t(halide_type_float, 16) == inputDataType) {
            BF162FLOAT<<<block_num, threads_num>>>((int16_t*)input, (half*)output, count);
            checkKernelErrors;
        } else {
            MNN_PRINT("Error: CUDABackend don't support cast form %d, %d to %d\n", inputDataType.code, inputDataType.bits, dstT);
        }
    } else {
        if (dstT == MNN::DataType_DT_INT32 && halide_type_of<float>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((float*)input, (int*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int32_t>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((int*)input, (float*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<uint8_t>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((uint8_t*)input, (float*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_of<int8_t>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((int8_t*)input, (float*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_INT8 && halide_type_of<float>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((float*)input, (int8_t*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_UINT8 && halide_type_of<float>() == inputDataType) {
            CASTMIDFLOAT<<<block_num, threads_num>>>((float*)input, (uint8_t*)output, count);
            checkKernelErrors;
        } else if (dstT == MNN::DataType_DT_FLOAT && halide_type_t(halide_type_float, 16) == inputDataType) {
            BF162FLOAT<<<block_num, threads_num>>>((int16_t*)input, (float*)output, count);
            checkKernelErrors;
        } else {
            MNN_PRINT("Error: CUDABackend don't support cast form %d, %d to %d\n", inputDataType.code, inputDataType.bits, dstT);
        }
    }
    checkKernelErrors;
    return NO_ERROR;
}

ErrorCode CastCreator::cast(const Tensor* input, const Tensor* output, ConvertType type,
    float scale, float zero, float min, float max, Backend* bn) {
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();
    auto input_addr = (void*)input->deviceId();
    auto output_addr = (void*)output->deviceId();

    auto count = CUDABackend::realSize(input);
    // MNN_PRINT("float2int8 size:%d scale:%f\n", count, scale);
    int block_num = runtime->blocks_num(count);
    int threads_num = runtime->threads_num();
    auto sfmt    = TensorUtils::getDescribe(input)->dimensionFormat;
    auto dfmt    = TensorUtils::getDescribe(output)->dimensionFormat;
    MNN_ASSERT(sfmt == dfmt);
    if(sfmt == MNN_DATA_FORMAT_NC4HW4) {
        auto area = input->batch() * input->height() * input->width();
        auto channel = input->channel();
        auto channelPackInt8 = UP_DIV(channel, INT8_PACK_NUMBER) * INT8_PACK_NUMBER;
        auto channelPackFloat = UP_DIV(channel, PACK_NUMBER) * PACK_NUMBER;

        if (type == FlOAT_TO_INT8) {
            DivModFast cpD(channelPackInt8);
            count = area * channelPackInt8;

            scale = (scale == 0.f ? 0.f : 1.f / scale);
            if (static_cast<CUDABackend*>(bn)->useFp16()) {
                FLOAT_2_INT8_CAST_PACK<<<block_num, threads_num>>>(count, (const half *)input_addr, (int8_t *)output_addr,\
                    scale, zero, max, min, channelPackFloat, channel, cpD);
                checkKernelErrors;
            } else {
                FLOAT_2_INT8_CAST_PACK<<<block_num, threads_num>>>(count, (const float *)input_addr, (int8_t *)output_addr,\
                    scale, zero, max, min, channelPackFloat, channel, cpD);
                checkKernelErrors;
            }
            return NO_ERROR;
        }
        if (type == INT8_TO_FlOAT) {
            DivModFast cpD(channelPackFloat);
            count = area * channelPackFloat;

            if (static_cast<CUDABackend*>(bn)->useFp16()) {
                INT8_2_FLOAT_CAST_PACK<<<block_num, threads_num>>>(count, (const int8_t *)input_addr, (half *)output_addr,\
                    scale, zero, channelPackInt8, channel, cpD);
                checkKernelErrors;
            } else {
                INT8_2_FLOAT_CAST_PACK<<<block_num, threads_num>>>(count, (const int8_t *)input_addr, (float *)output_addr,\
                    scale, zero, channelPackInt8, channel, cpD);
                checkKernelErrors;
            }
            return NO_ERROR;
        }
        MNN_ERROR("CUDA Don't support NC4HW4 cast type \n");

        return NO_ERROR;
    }

    if (type == FlOAT_TO_INT8) {
        scale = (scale == 0.f ? 0.f : 1.f / scale);
        if (static_cast<CUDABackend*>(bn)->useFp16()) {
            FLOAT_2_INT8_CAST<<<block_num, threads_num>>>(count, (const half *)input_addr, (int8_t *)output_addr,\
                scale, zero, max, min);
            checkKernelErrors;
        } else {
            FLOAT_2_INT8_CAST<<<block_num, threads_num>>>(count, (const float *)input_addr, (int8_t *)output_addr,\
                scale, zero, max, min);
            checkKernelErrors;
        }
        return NO_ERROR;
    }
    if (type == INT8_TO_FlOAT) {
        if (static_cast<CUDABackend*>(bn)->useFp16()) {
            INT8_2_FLOAT_CAST<<<block_num, threads_num>>>(count, (const int8_t *)input_addr, (half *)output_addr,\
                scale, zero);
            checkKernelErrors;
        } else {
            INT8_2_FLOAT_CAST<<<block_num, threads_num>>>(count, (const int8_t *)input_addr, (float *)output_addr,\
                scale, zero);
            checkKernelErrors;
        }
        return NO_ERROR;
    }
    MNN_ERROR("CUDA Don't support cast type \n");
    return NOT_SUPPORT;
}

ErrorCode CastCreator::cast(const Tensor* input, const Tensor* output, Backend* bn, ConvertType type) {
    auto quantAttr = TensorUtils::getDescribe(input)->quantAttr;
    if (quantAttr == nullptr) {
        MNN_ERROR("No quant info for CUDA Cast srcDataType:%d\n", static_cast<CUDABackend *>(bn)->getDataType(input));
        return INVALID_VALUE;
    }
    // MNN_PRINT("quant info for Cast %d\n", static_cast<const CUDABackend*>(bn)->getDataType(input));
    auto code = cast(input, output, type, quantAttr->scale, quantAttr->zero, quantAttr->min, quantAttr->max, bn);
    if (NO_ERROR != code) {
        MNN_ERROR("Error in CUDACast\n");
        return code;
    }
    return NO_ERROR;
}


Execution* CastCreator::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                            const MNN::Op* op, Backend* backend) const{
    return new CastExecution(backend, op->main_as_CastParam()->dstT());
}

CUDACreatorRegister<CastCreator> __CastExecution(OpType_Cast);
} // namespace CUDA
} // namespace MNN
