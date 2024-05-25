//
//  GridSampleExecution.cpp
//  MNN
//
//  Created by MNN on 2023/03/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GridSampleExecution.hpp"
#include "core/Macro.h"
#include <cuda_runtime.h>

namespace MNN {
namespace CUDA {
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)


inline __device__ float getPosition(float x, int range, bool alignCorners){
    float a = alignCorners == true ? 1.0f : 0.0f;
    float b = alignCorners == true ? 0.0f : 1.0f;
    return ((1.0f + x) * (range - a) - b) / 2.0f;
}

inline __device__ int CLAMP(int value, int minV, int maxV) {
    return min(max(value, minV), maxV);
}

inline __device__ int sample(int pos,
    int total, 
    BorderMode paddingMode){

    if (pos < 0 || pos >= total) {
        if(paddingMode == BorderMode_ZEROS) {
            return -1;
        }
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        pos = CLAMP(pos, 0, total - 1);
    }
    return pos;
}

template<typename T>
__global__ void GRID_SAMPLE_NEAREST(const int count, const T* input, const T* grid, T* output,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int channel,
    const int channel_pack,
    BorderMode paddingMode,
    bool alignCorners
    ) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp  = index % channel;
        int idx_nhw = index / channel;
        int idx_ow  = idx_nhw % output_width;
        int idx_nh  = idx_nhw / output_width;
        int idx_oh  = idx_nh % output_height;
        int idx_ob  = idx_nh / output_height;

        float pos_x = grid[idx_nhw * 2 + 0];
        float pos_y = grid[idx_nhw * 2 + 1];
        float in_grid_x = getPosition(pos_x, input_width, alignCorners);
        float in_grid_y = getPosition(pos_y, input_height, alignCorners);

        // get nearest point
        int in_pos_x = floor(in_grid_x + 0.5f);
        int in_pos_y = floor(in_grid_y + 0.5f);

        in_pos_x = sample(in_pos_x, input_width, paddingMode);
        in_pos_y = sample(in_pos_y, input_height, paddingMode);

        int dst_offset = ((idx_ob * output_height + idx_oh) * output_width + idx_ow) * channel_pack + idx_cp;
        if(in_pos_x == -1 || in_pos_y == -1) {
            output[dst_offset] = (T)0.0;
            continue;
        }

        output[dst_offset] = input[((idx_ob * input_height + in_pos_y) * input_width + in_pos_x) * channel_pack + idx_cp];
    }
}

template<typename T>
__global__ void GRID_SAMPLE_NEAREST_3D(const int count, const T* input, const T* grid, T* output,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int channel,
    const int channel_pack,
    BorderMode paddingMode,
    bool alignCorners
    ) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp  = index % channel;
        int idx_nhw = index / channel;
        int idx_ow  = idx_nhw % output_width;
        int idx_nh  = idx_nhw / output_width;
        int idx_oh  = idx_nh % output_height;
        int idx_obd = idx_nh / output_height;
        int idx_od  = idx_obd % output_depth;
        int idx_ob  = idx_obd / output_depth;

        float pos_x = grid[idx_nhw * 3 + 0];
        float pos_y = grid[idx_nhw * 3 + 1];
        float pos_z = grid[idx_nhw * 3 + 2];
        float in_grid_x = getPosition(pos_x, input_width, alignCorners);
        float in_grid_y = getPosition(pos_y, input_height, alignCorners);
        float in_grid_z = getPosition(pos_z, input_depth, alignCorners);

        // get nearest point
        int in_pos_x = floor(in_grid_x + 0.5f);
        int in_pos_y = floor(in_grid_y + 0.5f);
        int in_pos_z = floor(in_grid_z + 0.5f);

        in_pos_x = sample(in_pos_x, input_width, paddingMode);
        in_pos_y = sample(in_pos_y, input_height, paddingMode);
        in_pos_z = sample(in_pos_z, input_depth, paddingMode);

        int dst_offset = (((idx_ob * output_depth + idx_od) * output_height + idx_oh) * output_width + idx_ow) * channel_pack + idx_cp;
        if(in_pos_x == -1 || in_pos_y == -1 || in_pos_z == -1) {
            output[dst_offset] = (T)0.0;
            continue;
        }

        output[dst_offset] = input[(((idx_ob * input_depth + in_pos_z) * input_height + in_pos_y) * input_width + in_pos_x) * channel_pack + idx_cp];
    }
}

template<typename T>
__global__ void GRID_SAMPLE_BILINEAR(const int count, const T* input, const T* grid, T* output,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int channel,
    const int channel_pack,
    BorderMode paddingMode,
    bool alignCorners
    ) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp  = index % channel;
        int idx_nhw = index / channel;
        int idx_ow  = idx_nhw % output_width;
        int idx_nh  = idx_nhw / output_width;
        int idx_oh  = idx_nh % output_height;
        int idx_ob  = idx_nh / output_height;
        float pos_x = grid[idx_nhw * 2 + 0];
        float pos_y = grid[idx_nhw * 2 + 1];
        float in_grid_x = getPosition(pos_x, input_width, alignCorners);
        float in_grid_y = getPosition(pos_y, input_height, alignCorners);

        // get nearest point
        int in_pos_x0 = floor(in_grid_x);
        int in_pos_y0 = floor(in_grid_y);
        int in_pos_x1 = ceil(in_grid_x);
        int in_pos_y1 = ceil(in_grid_y);

        float x_weight = in_pos_x1 - in_grid_x;
        float y_weight = in_pos_y1 - in_grid_y;

        in_pos_x0 = sample(in_pos_x0, input_width, paddingMode);
        in_pos_y0 = sample(in_pos_y0, input_height, paddingMode);
        in_pos_x1 = sample(in_pos_x1, input_width, paddingMode);
        in_pos_y1 = sample(in_pos_y1, input_height, paddingMode);

        float in00 = (in_pos_y0 == -1 || in_pos_x0 == -1) ? 0.0 : (float)input[((idx_ob * input_height + in_pos_y0) * input_width + in_pos_x0) * channel_pack + idx_cp];
        float in01 = (in_pos_y0 == -1 || in_pos_x1 == -1) ? 0.0 : (float)input[((idx_ob * input_height + in_pos_y0) * input_width + in_pos_x1) * channel_pack + idx_cp];
        float in10 = (in_pos_y1 == -1 || in_pos_x0 == -1) ? 0.0 : (float)input[((idx_ob * input_height + in_pos_y1) * input_width + in_pos_x0) * channel_pack + idx_cp];
        float in11 = (in_pos_y1 == -1 || in_pos_x1 == -1) ? 0.0 : (float)input[((idx_ob * input_height + in_pos_y1) * input_width + in_pos_x1) * channel_pack + idx_cp];

        int dst_offset = ((idx_ob * output_height + idx_oh) * output_width + idx_ow) * channel_pack + idx_cp;
        output[dst_offset] = (T)(in00 * x_weight * y_weight + in01 * (1.0-x_weight) * y_weight + in10 * x_weight * (1.0-y_weight) + in11 * (1.0-x_weight) * (1.0-y_weight));
    }
}

template<typename T>
__global__ void GRID_SAMPLE_BILINEAR_3D(const int count, const T* input, const T* grid, T* output,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int channel,
    const int channel_pack,
    BorderMode paddingMode,
    bool alignCorners
    ) {
    CUDA_KERNEL_LOOP(index, count) {
        int idx_cp  = index % channel;
        int idx_nhw = index / channel;
        int idx_ow  = idx_nhw % output_width;
        int idx_nh  = idx_nhw / output_width;
        int idx_oh  = idx_nh % output_height;
        int idx_obd = idx_nh / output_height;
        int idx_od  = idx_obd % output_depth;
        int idx_ob  = idx_obd / output_depth;

        float pos_x = grid[idx_nhw * 3 + 0];
        float pos_y = grid[idx_nhw * 3 + 1];
        float pos_z = grid[idx_nhw * 3 + 2];

        float in_grid_x = getPosition(pos_x, input_width, alignCorners);
        float in_grid_y = getPosition(pos_y, input_height, alignCorners);
        float in_grid_z = getPosition(pos_z, input_depth, alignCorners);

        // get nearest point
        int in_pos_x0 = floor(in_grid_x);
        int in_pos_y0 = floor(in_grid_y);
        int in_pos_z0 = floor(in_grid_z);

        int in_pos_x1 = ceil(in_grid_x);
        int in_pos_y1 = ceil(in_grid_y);
        int in_pos_z1 = ceil(in_grid_z);

        float x_weight = in_pos_x1 - in_grid_x;
        float y_weight = in_pos_y1 - in_grid_y;
        float z_weight = in_pos_z1 - in_grid_z;

        in_pos_x0 = sample(in_pos_x0, input_width, paddingMode);
        in_pos_y0 = sample(in_pos_y0, input_height, paddingMode);
        in_pos_z0 = sample(in_pos_z0, input_depth, paddingMode);

        in_pos_x1 = sample(in_pos_x1, input_width, paddingMode);
        in_pos_y1 = sample(in_pos_y1, input_height, paddingMode);
        in_pos_z1 = sample(in_pos_z1, input_depth, paddingMode);

        float in000 = (in_pos_z0 == -1 || in_pos_y0 == -1 || in_pos_x0 == -1) ? 0.0 : (float)input[(((idx_ob * input_depth + in_pos_z0) * input_height + in_pos_y0) * input_width + in_pos_x0) * channel_pack + idx_cp];
        float in001 = (in_pos_z0 == -1 || in_pos_y0 == -1 || in_pos_x1 == -1) ? 0.0 : (float)input[(((idx_ob * input_depth + in_pos_z0) * input_height + in_pos_y0) * input_width + in_pos_x1) * channel_pack + idx_cp];
        float in010 = (in_pos_z0 == -1 || in_pos_y1 == -1 || in_pos_x0 == -1) ? 0.0 : (float)input[(((idx_ob * input_depth + in_pos_z0) * input_height + in_pos_y1) * input_width + in_pos_x0) * channel_pack + idx_cp];
        float in011 = (in_pos_z0 == -1 || in_pos_y1 == -1 || in_pos_x1 == -1) ? 0.0 : (float)input[(((idx_ob * input_depth + in_pos_z0) * input_height + in_pos_y1) * input_width + in_pos_x1) * channel_pack + idx_cp];

        float in100 = (in_pos_z1 == -1 || in_pos_y0 == -1 || in_pos_x0 == -1) ? 0.0 : (float)input[(((idx_ob * input_depth + in_pos_z1) * input_height + in_pos_y0) * input_width + in_pos_x0) * channel_pack + idx_cp];
        float in101 = (in_pos_z1 == -1 || in_pos_y0 == -1 || in_pos_x1 == -1) ? 0.0 : (float)input[(((idx_ob * input_depth + in_pos_z1) * input_height + in_pos_y0) * input_width + in_pos_x1) * channel_pack + idx_cp];
        float in110 = (in_pos_z1 == -1 || in_pos_y1 == -1 || in_pos_x0 == -1) ? 0.0 : (float)input[(((idx_ob * input_depth + in_pos_z1) * input_height + in_pos_y1) * input_width + in_pos_x0) * channel_pack + idx_cp];
        float in111 = (in_pos_z1 == -1 || in_pos_y1 == -1 || in_pos_x1 == -1) ? 0.0 : (float)input[(((idx_ob * input_depth + in_pos_z1) * input_height + in_pos_y1) * input_width + in_pos_x1) * channel_pack + idx_cp];
        int dst_offset = (((idx_ob * output_depth + idx_od) * output_height + idx_oh) * output_width + idx_ow) * channel_pack + idx_cp;

        output[dst_offset] = (T)(in000 * x_weight * y_weight * z_weight + in001 * (1.0-x_weight) * y_weight * z_weight + in010 * x_weight * (1.0-y_weight) * z_weight + in011 * (1.0-x_weight) * (1.0-y_weight) * z_weight + \
                            in100 * x_weight * y_weight * (1.0-z_weight) + in101 * (1.0-x_weight) * y_weight * (1.0-z_weight) + in110 * x_weight * (1.0-y_weight) * (1.0-z_weight) + in111 * (1.0-x_weight) * (1.0-y_weight) * (1.0-z_weight));
    }
}

GridSampleExecution::GridSampleExecution(Backend* backend, SampleMode mode, BorderMode paddingMode, bool alignCorners) : Execution(backend) {
    mMode = mode;
    mPaddingMode = paddingMode;
    mAlignCorners = alignCorners;
}
ErrorCode GridSampleExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    if (outputs[0]->dimensions() == 4) {
        mChannel = input->channel();
        mBatch   = input->batch();

        mInputHeight  = input->height();
        mInputWidth   = input->width();
        mOutputHeight = output->height();
        mOutputWidth  = output->width();
        mChannelPack  = UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER;
        mCount = mBatch* mOutputHeight * mOutputWidth * mChannel;
    } else {
        MNN_ASSERT(outputs[0]->dimensions() == 5);
        mChannel = input->buffer().dim[1].extent;
        mBatch   = input->buffer().dim[0].extent;
        mInputDepth   = input->buffer().dim[2].extent;
        mInputHeight  = input->buffer().dim[3].extent;
        mInputWidth   = input->buffer().dim[4].extent;

        mOutputDepth  = output->buffer().dim[2].extent;
        mOutputHeight = output->buffer().dim[3].extent;
        mOutputWidth  = output->buffer().dim[4].extent;
        mChannelPack  = UP_DIV(mChannel, PACK_NUMBER) * PACK_NUMBER;
        mCount = mBatch * mOutputDepth * mOutputHeight * mOutputWidth * mChannel;
    }
    // MNN_PRINT("GridSample: %d %d %d %d %d %d\n", mBatch, mInputHeight, mInputWidth, mOutputHeight, mOutputWidth, mChannel);
    return NO_ERROR;
}

ErrorCode GridSampleExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start GridSampleExecution onExecute...");
#endif
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    int block_num = runtime->blocks_num(mCount);
    int threads_num = runtime->threads_num();
    auto input_addr = (void*)inputs[0]->deviceId();
    auto grid_addr = (void*)inputs[1]->deviceId();
    auto output_addr = (void*)outputs[0]->deviceId();
    if (outputs[0]->dimensions() == 4) {
        if (static_cast<CUDABackend*>(backend())->useFp16()) {
            if(mMode == SampleMode_BILINEAR) {
                GRID_SAMPLE_BILINEAR<<<block_num, threads_num>>>(mCount, (const half*)input_addr, (const half*)grid_addr, (half*)output_addr, \
                    mInputHeight, mInputWidth, mOutputHeight, mOutputWidth, mChannel, mChannelPack, mPaddingMode, mAlignCorners);
                checkKernelErrors;
            } else {
                GRID_SAMPLE_NEAREST<<<block_num, threads_num>>>(mCount, (const half*)input_addr, (const half*)grid_addr, (half*)output_addr, \
                    mInputHeight, mInputWidth, mOutputHeight, mOutputWidth, mChannel, mChannelPack, mPaddingMode, mAlignCorners);
                checkKernelErrors;
            }
        } else {
            if(mMode == SampleMode_BILINEAR) {
                GRID_SAMPLE_BILINEAR<<<block_num, threads_num>>>(mCount, (const float*)input_addr, (const float*)grid_addr, (float*)output_addr, \
                    mInputHeight, mInputWidth, mOutputHeight, mOutputWidth, mChannel, mChannelPack, mPaddingMode, mAlignCorners);
                checkKernelErrors;
            } else {
                GRID_SAMPLE_NEAREST<<<block_num, threads_num>>>(mCount, (const float*)input_addr, (const float*)grid_addr, (float*)output_addr, \
                    mInputHeight, mInputWidth, mOutputHeight, mOutputWidth, mChannel, mChannelPack, mPaddingMode, mAlignCorners);
                checkKernelErrors;  
            }
        }
    } else {
        if (static_cast<CUDABackend*>(backend())->useFp16()) {
            if(mMode == SampleMode_BILINEAR) {
                GRID_SAMPLE_BILINEAR_3D<<<block_num, threads_num>>>(mCount, (const half*)input_addr, (const half*)grid_addr, (half*)output_addr, \
                    mInputDepth, mInputHeight, mInputWidth, mOutputDepth, mOutputHeight, mOutputWidth, mChannel, mChannelPack, mPaddingMode, mAlignCorners);
                checkKernelErrors;
            } else {
                GRID_SAMPLE_NEAREST_3D<<<block_num, threads_num>>>(mCount, (const half*)input_addr, (const half*)grid_addr, (half*)output_addr, \
                    mInputDepth, mInputHeight, mInputWidth, mOutputDepth, mOutputHeight, mOutputWidth, mChannel, mChannelPack, mPaddingMode, mAlignCorners);
                checkKernelErrors;
            }
        } else {
            if(mMode == SampleMode_BILINEAR) {
                GRID_SAMPLE_BILINEAR_3D<<<block_num, threads_num>>>(mCount, (const float*)input_addr, (const float*)grid_addr, (float*)output_addr, \
                    mInputDepth, mInputHeight, mInputWidth, mOutputDepth, mOutputHeight, mOutputWidth, mChannel, mChannelPack, mPaddingMode, mAlignCorners);
                checkKernelErrors;
            } else {
                GRID_SAMPLE_NEAREST_3D<<<block_num, threads_num>>>(mCount, (const float*)input_addr, (const float*)grid_addr, (float*)output_addr, \
                    mInputDepth, mInputHeight, mInputWidth, mOutputDepth, mOutputHeight, mOutputWidth, mChannel, mChannelPack, mPaddingMode, mAlignCorners);
                checkKernelErrors;  
            }
        }        
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("end GridSampleExecution onExecute...");
#endif
    return NO_ERROR;
}


class GridSampleCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto gridSampleParam = op->main_as_GridSample();
        auto mode = gridSampleParam->mode();
        auto paddingMode = gridSampleParam->paddingMode();
        auto alignCorners = gridSampleParam->alignCorners();

        // MNN_PRINT("GridSample config:%d %d %d\n\n", mode, paddingMode, alignCorners);
        return new GridSampleExecution(backend, mode, paddingMode, alignCorners);
    }
};

CUDACreatorRegister<GridSampleCreator> __GridSampleExecution(OpType_GridSample);
} // namespace CUDA
} // namespace MNN
