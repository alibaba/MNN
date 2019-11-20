#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void float_to_int8(
                      __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
                      __read_only image2d_t input, __global char* output_ptr, __global FLOAT4* scale_ptr, __private const int height, __private const int width) {
    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);
    
    if (channel_block_idx < global_size_dim0 && w < global_size_dim1 && hb < global_size_dim2) {
        const int pos  = mad24(channel_block_idx, width, w);
        FLOAT4 in  = RI_F(input, SAMPLER, (int2)(pos, hb));

        FLOAT4 scale = vload4(channel_block_idx, (__global FLOAT *)scale_ptr);

        FLOAT4 result_float = in * scale;
        int4 result_int = convert_int4_rte(result_float);

        char4 out = convert_char4_sat(result_int);

        int index = channel_block_idx*height*width + hb*width + w;
        vstore4(out, index, output_ptr);
    }
    
}
