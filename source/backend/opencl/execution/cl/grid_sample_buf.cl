#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

enum BorderMode {
  BorderMode_ZEROS = 0,
  BorderMode_CLAMP = 1,
  BorderMode_REFLECTION = 2,
  BorderMode_MIN = BorderMode_ZEROS,
  BorderMode_MAX = BorderMode_REFLECTION
};

float getPosition(float x, int range, int alignCorners){
    float a = alignCorners == 1? 1.0f : 0.0f;
    float b = alignCorners == 1? 0.0f : 1.0f;
    return ((1.0f + x) * (range - a) - b) / 2.0f;
}

static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

COMPUTE_FLOAT4 sample(int h, int w,
              const int offset_base, 
              __global const FLOAT *buffer, 
              int height, int width, 
              enum BorderMode paddingMode){

    if (h < 0 || h >= height || w < 0 || w >= width) {
        if(paddingMode == BorderMode_ZEROS)
        {
            return 0.0f;
        }
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        h = CLAMP(h, 0, height - 1);
        w = CLAMP(w, 0, width - 1);
    }
    int offset = (offset_base + h) * width + w;
    return CONVERT_COMPUTE_FLOAT4(vload4(offset, buffer));
}

COMPUTE_FLOAT4 sample3d(int d, int h, int w,
              const int offset_base,
              __global const FLOAT *buffer,
              int depth, int height, int width,
              enum BorderMode paddingMode){

    if (d < 0 || d >= depth || h < 0 || h >= height || w < 0 || w >= width) {
        if(paddingMode == BorderMode_ZEROS)
        {
            return 0.0f;
        }
        d = CLAMP(d, 0, depth - 1);
        h = CLAMP(h, 0, height - 1);
        w = CLAMP(w, 0, width - 1);
    }
    int offset = ((offset_base + d) * height + h) * width + w;
    return CONVERT_COMPUTE_FLOAT4(vload4(offset, buffer));
}

__kernel void nearest_buf(GLOBAL_SIZE_3_DIMS  __global const FLOAT* input,
                        __global const FLOAT* grid,
                        __global FLOAT* output,
                        __private const int input_height,
                        __private const int input_width,
                        __private const int output_height,
                        __private const int output_width,
                        __private const int batch,
                        __private const enum BorderMode paddingMode,
                        __private const int alignCorners){
    
    const int output_channel_block_idx      = get_global_id(0);
    const int output_width_block_idx        = get_global_id(1);
    const int output_batch_height_block_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_block_idx, output_width_block_idx, output_batch_height_block_idx);

    const int output_batch_idx  = output_batch_height_block_idx / output_height;
    const int output_height_idx = output_batch_height_block_idx % output_height;
    // grid data format has been converted from nchw to nc4hw4
    /*    
                                (x1,x1,x1,x1) (y1,y2,y3,y4) 
                                .                         . 
                                .                         . slice
        (x1,y1)...(xn,y1)       .                         . 
        .               .       (xn,xn,xn,xn) (y1,y2,y3,y4)
        .               .  <->  ---------------------------
        .               .       (x1,x1,x1,x1) (y5,y6,y7,y8)
        (x1,ym)...(xn,ym)       .                         .
                                .                         . slice
                                .                         .
                                (xn,xn,xn,xn) (y5,y6,y7,y8)
                                ---------------------------
    */
    // output_width_block_idx means gird y offset, 2 means grid width
    const int grid_offset = (output_batch_idx * output_height + output_height_idx) * output_width + output_width_block_idx;
    COMPUTE_FLOAT2 grid_xy = CONVERT_COMPUTE_FLOAT2(vload2(grid_offset, grid));

    // get grid x,y
    const float x = (float)grid_xy.x;
    const float y = (float)grid_xy.y;

    // convert grid x,y to input x,y coordinate range
    float in_grid_x = getPosition(x, input_width, alignCorners);
    float in_grid_y = getPosition(y, input_height, alignCorners);

    // get nearest point
    int nw = floor(in_grid_x + 0.5f);
    int nh = floor(in_grid_y + 0.5f);

    const int inp_offset_base = (output_batch_idx + output_channel_block_idx * batch) * input_height;
    COMPUTE_FLOAT4 value = sample(nh, nw, inp_offset_base, input, input_height, input_width, paddingMode);

    const int output_offset = ((output_batch_idx + output_channel_block_idx * batch) * output_height + output_height_idx) * output_width + output_width_block_idx;
    vstore4(CONVERT_FLOAT4(value), output_offset, output);
}

__kernel void bilinear_buf(GLOBAL_SIZE_3_DIMS  __global const FLOAT* input,
                        __global const FLOAT* grid,
                        __global FLOAT* output,
                        __private const int input_height,
                        __private const int input_width,
                        __private const int output_height,
                        __private const int output_width,
                        __private const int batch,
                        __private const enum BorderMode paddingMode,
                        __private const int alignCorners){

    const int output_channel_block_idx      = get_global_id(0);
    const int output_width_block_idx        = get_global_id(1);
    const int output_batch_height_block_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_block_idx, output_width_block_idx, output_batch_height_block_idx);

    const int output_batch_idx  = output_batch_height_block_idx / output_height;
    const int output_height_idx = output_batch_height_block_idx % output_height;

    // output_width_block_idx means gird y offset, 2 means grid width
    const int grid_offset = (output_batch_idx * output_height + output_height_idx) * output_width + output_width_block_idx;
    COMPUTE_FLOAT2 grid_xy = CONVERT_COMPUTE_FLOAT2(vload2(grid_offset, grid));

    
    // get grid x,y
    const float x = (float)grid_xy.x;
    const float y = (float)grid_xy.y;

    // convert grid x,y to input x,y coordinate range
    float in_grid_x = getPosition(x, input_width, alignCorners);
    float in_grid_y = getPosition(y, input_height, alignCorners);

    int in_h0 = floor(in_grid_y);
    int in_w0 = floor(in_grid_x);
    int in_h1 = ceil(in_grid_y);
    int in_w1 = ceil(in_grid_x);

    float x_weight = in_w1 - in_grid_x;
    float y_weight = in_h1 - in_grid_y;

    // bilinear interpolation
    const int inp_offset_base = (output_batch_idx + output_channel_block_idx * batch) * input_height;
    COMPUTE_FLOAT4 i00 = sample(in_h0, in_w0, inp_offset_base, input, input_height, input_width, paddingMode);
    COMPUTE_FLOAT4 i01 = sample(in_h0, in_w1, inp_offset_base, input, input_height, input_width, paddingMode);
    COMPUTE_FLOAT4 i10 = sample(in_h1, in_w0, inp_offset_base, input, input_height, input_width, paddingMode);
    COMPUTE_FLOAT4 i11 = sample(in_h1, in_w1, inp_offset_base, input, input_height, input_width, paddingMode);

    COMPUTE_FLOAT4 value = CONVERT_COMPUTE_FLOAT4(((COMPUTE_FLOAT4)x_weight * CONVERT_COMPUTE_FLOAT4(i00)  + (COMPUTE_FLOAT4)(1.0f - x_weight) * CONVERT_COMPUTE_FLOAT4(i01)) * (COMPUTE_FLOAT4)y_weight  +
                    ((COMPUTE_FLOAT4)x_weight * CONVERT_COMPUTE_FLOAT4(i10)  + (COMPUTE_FLOAT4)(1.0f - x_weight) * CONVERT_COMPUTE_FLOAT4(i11)) * (COMPUTE_FLOAT4)(1.0f- y_weight));
    
    const int output_offset = ((output_batch_idx + output_channel_block_idx * batch) * output_height + output_height_idx) * output_width + output_width_block_idx;
    vstore4(CONVERT_FLOAT4(value), output_offset, output);
}
__kernel void nearest5d_buf(GLOBAL_SIZE_3_DIMS  __global const FLOAT* input,
                        __global const FLOAT* grid,
                        __global FLOAT* output,
                        __private const int input_height,
                        __private const int input_width,
                        __private const int input_depth,
                        __private const int output_height,
                        __private const int output_width,
                        __private const int output_depth,
                        __private const int batch,
                        __private const enum BorderMode paddingMode,
                        __private const int alignCorners){
    
    const int output_channel_depth_idx      = get_global_id(0);
    const int output_width_block_idx        = get_global_id(1);
    const int output_batch_height_block_idx = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(output_channel_depth_idx, output_width_block_idx, output_batch_height_block_idx);
    
    const int output_channel_idx = output_channel_depth_idx / output_depth;
    const int output_depth_idx = output_channel_depth_idx % output_depth;
    const int output_batch_idx  = output_batch_height_block_idx / output_height;
    const int output_height_idx = output_batch_height_block_idx % output_height;
    
    const int grid_offset = ((output_batch_idx * output_depth + output_depth_idx) * output_height + output_height_idx) * output_width + output_width_block_idx;
    float3 grid_xyz = convert_float3(vload3(grid_offset, grid));

    const float x = grid_xyz.x;
    const float y = grid_xyz.y;
    const float z = grid_xyz.z;

    float in_grid_x = getPosition(x, input_width, alignCorners);
    float in_grid_y = getPosition(y, input_height, alignCorners);
    float in_grid_z = getPosition(z, input_depth, alignCorners);

    // get nearest point
    int nw = floor(in_grid_x + 0.5f);
    int nh = floor(in_grid_y + 0.5f);
    int nd = floor(in_grid_z + 0.5f);

    const int inp_offset_base = (output_batch_idx + output_channel_idx * batch) * input_depth;
    COMPUTE_FLOAT4 value = sample3d(nd, nh, nw, inp_offset_base, input, input_depth, input_height, input_width, paddingMode);

    const int output_offset = (((output_batch_idx + output_channel_idx * batch) * output_depth + output_depth_idx) * output_height + output_height_idx) * output_width + output_width_block_idx;
    vstore4(CONVERT_FLOAT4(value), output_offset, output);
}

__kernel void bilinear5d_buf(GLOBAL_SIZE_3_DIMS
                        __global const FLOAT* input,
                        __global const FLOAT* grid,
                        __global FLOAT* output,
                        __private const int input_height,
                        __private const int input_width,
                        __private const int input_depth,
                        __private const int output_height,
                        __private const int output_width,
                        __private const int output_depth,
                        __private const int batch,
                        __private const enum BorderMode paddingMode,
                        __private const int alignCorners){

    const int output_channel_depth_idx      = get_global_id(0);
    const int output_width_block_idx        = get_global_id(1);
    const int output_batch_height_block_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_depth_idx, output_width_block_idx, output_batch_height_block_idx);
    
    const int output_channel_idx = output_channel_depth_idx / output_depth;
    const int output_depth_idx = output_channel_depth_idx % output_depth;
    const int output_batch_idx  = output_batch_height_block_idx / output_height;
    const int output_height_idx = output_batch_height_block_idx % output_height;
    
    const int grid_offset = ((output_batch_idx * output_depth + output_depth_idx) * output_height + output_height_idx) * output_width + output_width_block_idx;
    float3 grid_xyz = convert_float3(vload3(grid_offset, grid));

    
    // get grid x,y
    const float x = grid_xyz.x;
    const float y = grid_xyz.y;
    const float z = grid_xyz.z;

    float in_grid_x = getPosition(x, input_width, alignCorners);
    float in_grid_y = getPosition(y, input_height, alignCorners);
    float in_grid_z = getPosition(z, input_depth, alignCorners);

    int in_d0 = floor(in_grid_z);
    int in_h0 = floor(in_grid_y);
    int in_w0 = floor(in_grid_x);
    int in_d1 = ceil(in_grid_z);
    int in_h1 = ceil(in_grid_y);
    int in_w1 = ceil(in_grid_x);
    
    float x_weight0 = in_grid_x - in_w0;
    float x_weight1 = 1 - x_weight0;
    float y_weight0 = in_grid_y - in_h0;
    float y_weight1 = 1 - y_weight0;
    float z_weight0 = in_grid_z - in_d0;
    float z_weight1 = 1 - z_weight0;

    // bilinear interpolation
    const int inp_offset_base = (output_batch_idx + output_channel_idx * batch) * input_depth;
    COMPUTE_FLOAT4 i000 = sample3d(in_d0, in_h0, in_w0, inp_offset_base, input, input_depth, input_height, input_width, paddingMode);
    COMPUTE_FLOAT4 i001 = sample3d(in_d0, in_h0, in_w1, inp_offset_base, input, input_depth, input_height, input_width, paddingMode);
    COMPUTE_FLOAT4 i010 = sample3d(in_d0, in_h1, in_w0, inp_offset_base, input, input_depth, input_height, input_width, paddingMode);
    COMPUTE_FLOAT4 i011 = sample3d(in_d0, in_h1, in_w1, inp_offset_base, input, input_depth, input_height, input_width, paddingMode);
    COMPUTE_FLOAT4 i100 = sample3d(in_d1, in_h0, in_w0, inp_offset_base, input, input_depth, input_height, input_width, paddingMode);
    COMPUTE_FLOAT4 i101 = sample3d(in_d1, in_h0, in_w1, inp_offset_base, input, input_depth, input_height, input_width, paddingMode);
    COMPUTE_FLOAT4 i110 = sample3d(in_d1, in_h1, in_w0, inp_offset_base, input, input_depth, input_height, input_width, paddingMode);
    COMPUTE_FLOAT4 i111 = sample3d(in_d1, in_h1, in_w1, inp_offset_base, input, input_depth, input_height, input_width, paddingMode);
    
    
    COMPUTE_FLOAT4 i00 = (COMPUTE_FLOAT4)(x_weight1) * i000  + (COMPUTE_FLOAT4)(x_weight0) * i001;
    COMPUTE_FLOAT4 i01 = (COMPUTE_FLOAT4)(x_weight1) * i010  + (COMPUTE_FLOAT4)(x_weight0) * i011;
    COMPUTE_FLOAT4 i10 = (COMPUTE_FLOAT4)(x_weight1) * i100  + (COMPUTE_FLOAT4)(x_weight0) * i101;
    COMPUTE_FLOAT4 i11 = (COMPUTE_FLOAT4)(x_weight1) * i110  + (COMPUTE_FLOAT4)(x_weight0) * i111;
    
    COMPUTE_FLOAT4 i0 = (COMPUTE_FLOAT4)(y_weight1) * i00  + (COMPUTE_FLOAT4)(y_weight0) * i01;
    COMPUTE_FLOAT4 i1 = (COMPUTE_FLOAT4)(y_weight1) * i10  + (COMPUTE_FLOAT4)(y_weight0) * i11;
    COMPUTE_FLOAT4 interp = (COMPUTE_FLOAT4)(z_weight1) * i0 + (COMPUTE_FLOAT4)(z_weight0) * i1;
    
    const int output_offset = (((output_batch_idx + output_channel_idx * batch) * output_depth + output_depth_idx) * output_height + output_height_idx) * output_width + output_width_block_idx;
    vstore4(CONVERT_FLOAT4(interp), output_offset, output);
}
