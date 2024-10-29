#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

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
    return ((1 + x) * (range - a) - b) / 2.0f;
}

static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

FLOAT4 sample(int h, int w, 
              const int w_offset_base, 
              const int h_offset_base,
              __read_only image2d_t tmp,
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
    return RI_F(tmp, SAMPLER, (int2)(w_offset_base + w, h_offset_base + h));
}


FLOAT4 sample3d(int d, int h, int w,
              const int x_offset_base,
              const int y_offset_base,
              __read_only image2d_t tmp,
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
    return RI_F(tmp, SAMPLER, (int2)(x_offset_base + h * width + w, y_offset_base + d));
}


__kernel void nearest(GLOBAL_SIZE_3_DIMS  __read_only image2d_t input,
                        __read_only image2d_t grid,
                        __write_only image2d_t output,
                        __private const int input_height,
                        __private const int input_width,
                        __private const int output_height,
                        __private const int output_width,
                        __private const enum BorderMode paddingMode,
                        __private const int alignCorners
                        ){

    const int output_channel_block_idx      = get_global_id(0);
    const int output_width_block_idx        = get_global_id(1);
    const int output_batch_height_block_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_block_idx, output_width_block_idx, output_batch_height_block_idx);

    const int output_batch_idx  = output_batch_height_block_idx / output_height;
    const int output_height_idx = output_batch_height_block_idx % output_height;

    // grid data format has been converted from nchw to nc4hw4
    /*                                      slice                       slice
        (x1,y1)...(xn,y1)       (x1,x1,x1,x1) (y1,y2,y3,y4) | (x1,x1,x1,x1) (y5,y6,y7,y8) | ... 
        .               .       .                         . | .                         . |
        .               .  <->  .                         . | .                         . |
        .               .       .                         . | .                         . |
        (x1,ym)...(xn,ym)       (xn,xn,xn,xn) (y1,y2,y3,y4) | (xn,xn,xn,xn) (y5,y6,y7,y8) | ...
    */
    const int slice = output_height_idx / 4;
    const int grid_w_offset = 0;
    const int grid_h_offset = mad24(output_batch_idx, output_width, output_width_block_idx);
    
    FLOAT4 grid_x = RI_F(grid, SAMPLER, (int2)(grid_w_offset + 2 * slice, grid_h_offset));
    FLOAT4 grid_y = RI_F(grid, SAMPLER, (int2)(grid_w_offset + 1 + 2 * slice, grid_h_offset));

    const float arr[8] = {grid_x.x, grid_y.x, grid_x.y, grid_y.y, grid_x.z, grid_y.z, grid_x.w, grid_y.w};
    
    // get grid x,y
    const int arr_offset = output_height_idx % 4;
    const float x = arr[2 * arr_offset];
    const float y = arr[2 * arr_offset + 1];

    // convert grid x,y to input coordinate range
    float in_grid_x = getPosition(x, input_width, alignCorners);
    float in_grid_y = getPosition(y, input_height, alignCorners);

    // get nearest point
    int nw = floor(in_grid_x + 0.5f);
    int nh = floor(in_grid_y + 0.5f);

    const int inp_w_offset = mul24(output_channel_block_idx, input_width);
    const int inp_h_offset = mul24(output_batch_idx, input_height);
    FLOAT4 value = sample(nh, nw, inp_w_offset, inp_h_offset, input, input_height, input_width, paddingMode);

    const int output_w_offset = mad24(output_channel_block_idx, output_width, output_width_block_idx);
    const int output_h_offset = mad24(output_batch_idx, output_height, output_height_idx);
    WI_F(output, (int2)(output_w_offset, output_h_offset), value);
}

__kernel void bilinear(GLOBAL_SIZE_3_DIMS  __read_only image2d_t input,
                        __read_only image2d_t grid,
                        __write_only image2d_t output,
                        __private const int input_height,
                        __private const int input_width,
                        __private const int output_height,
                        __private const int output_width,
                        __private const enum BorderMode paddingMode,
                        __private const int alignCorners
                        ){
    const int output_channel_block_idx      = get_global_id(0);
    const int output_width_block_idx        = get_global_id(1);
    const int output_batch_height_block_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_block_idx, output_width_block_idx, output_batch_height_block_idx);

    const int output_batch_idx  = output_batch_height_block_idx / output_height;
    const int output_height_idx = output_batch_height_block_idx % output_height;

    // get grid idx
    const int slice = output_height_idx / 4;
    const int grid_w_offset = 0;
    const int grid_h_offset = mad24(output_batch_idx, output_width, output_width_block_idx);
    
    FLOAT4 grid_x = RI_F(grid, SAMPLER, (int2)(grid_w_offset + 2 * slice, grid_h_offset));
    FLOAT4 grid_y = RI_F(grid, SAMPLER, (int2)(grid_w_offset + 1 + 2 * slice, grid_h_offset));

    const float arr[8] = {grid_x.x, grid_y.x, grid_x.y, grid_y.y, grid_x.z, grid_y.z, grid_x.w, grid_y.w};
    
    // get grid x,y
    const int arr_offset = output_height_idx % 4;
    const float x = arr[2 * arr_offset];
    const float y = arr[2 * arr_offset + 1];

    // convert grid x,y to input coordinate range
    float in_grid_x = getPosition(x, input_width, alignCorners);
    float in_grid_y = getPosition(y, input_height, alignCorners);

    int in_h0 = floor(in_grid_y);
    int in_w0 = floor(in_grid_x);
    int in_h1 = ceil(in_grid_y);
    int in_w1 = ceil(in_grid_x);

    float x_weight = in_w1 - in_grid_x;
    float y_weight = in_h1 - in_grid_y;

    const int inp_w_offset = mul24(output_channel_block_idx, input_width);
    const int inp_h_offset = mul24(output_batch_idx, input_height);
    FLOAT4 i00 = sample(in_h0, in_w0, inp_w_offset,inp_h_offset, input, input_height, input_width, paddingMode);
    FLOAT4 i01 = sample(in_h0, in_w1, inp_w_offset,inp_h_offset, input, input_height, input_width, paddingMode);
    FLOAT4 i10 = sample(in_h1, in_w0, inp_w_offset,inp_h_offset, input, input_height, input_width, paddingMode);
    FLOAT4 i11 = sample(in_h1, in_w1, inp_w_offset,inp_h_offset, input, input_height, input_width, paddingMode);

    // bilinear interpolation
    FLOAT4 value = CONVERT_FLOAT4(((FLOAT4)x_weight * CONVERT_FLOAT4(i00)  + (FLOAT4)(1.0f - x_weight) * CONVERT_FLOAT4(i01)) * (FLOAT4)y_weight  +
                    ((FLOAT4)x_weight * CONVERT_FLOAT4(i10)  + (FLOAT4)(1.0f - x_weight) * CONVERT_FLOAT4(i11)) * (FLOAT4)(1.0f- y_weight));

    const int output_w_offset = mad24(output_channel_block_idx, output_width, output_width_block_idx);
    const int output_h_offset = mad24(output_batch_idx, output_height, output_height_idx);
    WI_F(output, (int2)(output_w_offset, output_h_offset), value);
}

__kernel void nearest5d(GLOBAL_SIZE_3_DIMS
                        __read_only image2d_t input,
                        __read_only image2d_t grid,
                        __write_only image2d_t output,
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
    
    // get grid idx
    const int grid_w_offset = (output_depth_idx / 4) * output_width * 3 + output_width_block_idx * 3;
    const int grid_h_offset = mad24(output_batch_idx, output_height, output_height_idx);
    
    FLOAT4 grid_x = RI_F(grid, SAMPLER, (int2)(grid_w_offset, grid_h_offset));
    FLOAT4 grid_y = RI_F(grid, SAMPLER, (int2)(grid_w_offset + 1, grid_h_offset));
    FLOAT4 grid_z = RI_F(grid, SAMPLER, (int2)(grid_w_offset + 2, grid_h_offset));

    const float arr[12] = {grid_x.x, grid_y.x, grid_z.x, grid_x.y, grid_y.y, grid_z.y, grid_x.z, grid_y.z, grid_z.z, grid_x.w, grid_y.w, grid_z.w};
    
    // get grid x,y
    const int arr_offset = output_depth_idx % 4;
    const float x = arr[3 * arr_offset];
    const float y = arr[3 * arr_offset + 1];
    const float z = arr[3 * arr_offset + 2];

    float in_grid_x = getPosition(x, input_width, alignCorners);
    float in_grid_y = getPosition(y, input_height, alignCorners);
    float in_grid_z = getPosition(z, input_depth, alignCorners);

    // get nearest point
    int nw = floor(in_grid_x + 0.5f);
    int nh = floor(in_grid_y + 0.5f);
    int nd = floor(in_grid_z + 0.5f);
    
    const int inp_w_offset = mul24(output_channel_idx, input_width * input_height);
    const int inp_h_offset = mul24(output_batch_idx, input_depth);
    FLOAT4 value = sample3d(nd, nh, nw, inp_w_offset, inp_h_offset, input, input_depth, input_height, input_width, paddingMode);
    
    const int output_w_offset = output_channel_idx * output_width * output_height + output_height_idx * output_width + output_width_block_idx;
    const int output_h_offset = mad24(output_batch_idx, output_depth, output_depth_idx);
    WI_F(output, (int2)(output_w_offset, output_h_offset), value);
}

__kernel void bilinear5d(GLOBAL_SIZE_3_DIMS
                        __read_only image2d_t input,
                        __read_only image2d_t grid,
                        __write_only image2d_t output,
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
    
    // get grid idx
    const int grid_w_offset = (output_depth_idx / 4) * output_width * 3 + output_width_block_idx * 3;
    const int grid_h_offset = mad24(output_batch_idx, output_height, output_height_idx);
    
    FLOAT4 grid_x = RI_F(grid, SAMPLER, (int2)(grid_w_offset, grid_h_offset));
    FLOAT4 grid_y = RI_F(grid, SAMPLER, (int2)(grid_w_offset + 1, grid_h_offset));
    FLOAT4 grid_z = RI_F(grid, SAMPLER, (int2)(grid_w_offset + 2, grid_h_offset));

    const float arr[12] = {grid_x.x, grid_y.x, grid_z.x, grid_x.y, grid_y.y, grid_z.y, grid_x.z, grid_y.z, grid_z.z, grid_x.w, grid_y.w, grid_z.w};
    
    // get grid x,y
    const int arr_offset = output_depth_idx % 4;
    const float x = arr[3 * arr_offset];
    const float y = arr[3 * arr_offset + 1];
    const float z = arr[3 * arr_offset + 2];

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
    const int inp_x_offset = mul24(output_channel_idx, input_width * input_height);
    const int inp_y_offset = mul24(output_batch_idx, input_depth);
    FLOAT4 i000 = sample3d(in_d0, in_h0, in_w0, inp_x_offset, inp_y_offset, input, input_depth, input_height, input_width, paddingMode);
    FLOAT4 i001 = sample3d(in_d0, in_h0, in_w1, inp_x_offset, inp_y_offset, input, input_depth, input_height, input_width, paddingMode);
    FLOAT4 i010 = sample3d(in_d0, in_h1, in_w0, inp_x_offset, inp_y_offset, input, input_depth, input_height, input_width, paddingMode);
    FLOAT4 i011 = sample3d(in_d0, in_h1, in_w1, inp_x_offset, inp_y_offset, input, input_depth, input_height, input_width, paddingMode);
    FLOAT4 i100 = sample3d(in_d1, in_h0, in_w0, inp_x_offset, inp_y_offset, input, input_depth, input_height, input_width, paddingMode);
    FLOAT4 i101 = sample3d(in_d1, in_h0, in_w1, inp_x_offset, inp_y_offset, input, input_depth, input_height, input_width, paddingMode);
    FLOAT4 i110 = sample3d(in_d1, in_h1, in_w0, inp_x_offset, inp_y_offset, input, input_depth, input_height, input_width, paddingMode);
    FLOAT4 i111 = sample3d(in_d1, in_h1, in_w1, inp_x_offset, inp_y_offset, input, input_depth, input_height, input_width, paddingMode);
    
    
    FLOAT4 i00 = (FLOAT4)(x_weight1) * i000  + (FLOAT4)(x_weight0) * i001;
    FLOAT4 i01 = (FLOAT4)(x_weight1) * i010  + (FLOAT4)(x_weight0) * i011;
    FLOAT4 i10 = (FLOAT4)(x_weight1) * i100  + (FLOAT4)(x_weight0) * i101;
    FLOAT4 i11 = (FLOAT4)(x_weight1) * i110  + (FLOAT4)(x_weight0) * i111;
    
    FLOAT4 i0 = (FLOAT4)(y_weight1) * i00  + (FLOAT4)(y_weight0) * i01;
    FLOAT4 i1 = (FLOAT4)(y_weight1) * i10  + (FLOAT4)(y_weight0) * i11;
    FLOAT4 interp = (FLOAT4)(z_weight1) * i0 + (FLOAT4)(z_weight0) * i1;
    const int output_w_offset = output_channel_idx * output_width * output_height + output_height_idx * output_width + output_width_block_idx;
    const int output_h_offset = mad24(output_batch_idx, output_depth, output_depth_idx);

    WI_F(output, (int2)(output_w_offset, output_h_offset), interp);
}
