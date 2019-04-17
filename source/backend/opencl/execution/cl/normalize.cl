#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void normalize_kernel(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t scale,
                               __private const float eps, __private const int channels,
                               __private const int remain_channels, __write_only image2d_t output) {
    const int chan_blk_idx = get_global_id(0);
    const int width_idx    = get_global_id(1);
    const int hb_idx       = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(chan_blk_idx, width_idx, hb_idx);
    int chan_blks;
    if (0 == remain_channels) {
        chan_blks = global_size_dim0;
    } else {
        chan_blks = global_size_dim0 - 1;
    }

    const int width = global_size_dim1;

    int pos           = width_idx;
    float sum     = 0;
    float4 scale_ = 0;
    float4 data;
    for (short i = 0; i < chan_blks; ++i) {
        data = read_imagef(input, SAMPLER, (int2)(pos, hb_idx));
        sum += data.x * data.x;
        sum += data.y * data.y;
        sum += data.z * data.z;
        sum += data.w * data.w;
        pos += width;
    }

    data = read_imagef(input, SAMPLER, (int2)(pos, hb_idx));
    switch (remain_channels) {
        case 1:
            sum += data.x * data.x;
            sum += data.y * data.y;
            sum += data.z * data.z;
        case 2:
            sum += data.x * data.x;
            sum += data.y * data.y;
        case 3:
            sum += data.x * data.x;
    }

    sum = 1.0f / sqrt(sum + eps);

    pos = mad24(chan_blk_idx, width, width_idx);

    data   = read_imagef(input, SAMPLER, (int2)(pos, hb_idx));
    scale_ = read_imagef(scale, SAMPLER, (int2)(chan_blk_idx, 0));

    float4 sum_vec = (float4)(sum);
    data               = data * sum_vec * scale_;

    write_imagef(output, (int2)(pos, hb_idx), data);
}
