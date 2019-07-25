#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void convert(
                      __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
                      __read_only image2d_t input, __write_only image2d_t output) {
    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);
    if (channel_block_idx < global_size_dim0 && w < global_size_dim1 && hb < global_size_dim2) {
        const int width = global_size_dim1;
        
        const int pos  = mad24(channel_block_idx, width, w);
        FLOAT4 in  = RI_F(input, SAMPLER, (int2)(pos, hb));
        FLOAT4 out = in;
        WI_F(output, (int2)(pos, hb), out);
    }
    
}
