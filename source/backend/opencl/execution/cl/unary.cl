#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void unary(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __write_only image2d_t output) {
    const int channel_block_idx = get_global_id(0);
    const int w                 = get_global_id(1);
    const int hb                = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(channel_block_idx, w, hb);
    const int width = global_size_dim1;

    const int pos  = mad24(channel_block_idx, width, w);
    float4 in  = read_imagef(input, SAMPLER, (int2)(pos, hb));
    float4 out = OPERATOR;
    write_imagef(output, (int2)(pos, hb), out);
}
