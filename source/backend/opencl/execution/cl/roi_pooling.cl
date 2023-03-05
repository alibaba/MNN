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

#define MIN_VALUE -FLT_MAX

// Supported data type: half/float
__kernel void roi_pooling(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t roi,
                          __private const int in_height, __private const int in_width, __private const int in_batch,
                          __private const int out_height, __private const int out_width, __private const float spatial_scale,
                          __write_only image2d_t output) {
    const int out_channel_idx = get_global_id(0);
    const int out_width_idx   = get_global_id(1);
    const int out_hb_idx      = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(out_channel_idx, out_width_idx, out_hb_idx);

    const int roi_batch_idx  = out_hb_idx / out_height;
    const int out_height_idx = out_hb_idx % out_height;

#if defined ROI_C1H1W5
    FLOAT4 roi_0 = RI_F(roi, SAMPLER, (int2)(0, roi_batch_idx));

    int input_batch = roi_0.x;
    if(input_batch >= in_batch){
        return;
    }
    FLOAT4 roi_1 = RI_F(roi, SAMPLER, (int2)(1, roi_batch_idx));
    FLOAT4 roi_2 = RI_F(roi, SAMPLER, (int2)(2, roi_batch_idx));
    FLOAT4 roi_3 = RI_F(roi, SAMPLER, (int2)(3, roi_batch_idx));
    FLOAT4 roi_4 = RI_F(roi, SAMPLER, (int2)(4, roi_batch_idx));
    int x1          = round(roi_1.x * spatial_scale);
    int y1          = round(roi_2.x * spatial_scale);
    int x2          = round(roi_3.x * spatial_scale);
    int y2          = round(roi_4.x * spatial_scale);
#elif defined ROI_C5H1W1
    FLOAT4 roi_0 = RI_F(roi, SAMPLER, (int2)(0, roi_batch_idx));

    int input_batch = roi_0.x;
    if(input_batch >= in_batch){
        return;
    }
    FLOAT4 roi_1 = RI_F(roi, SAMPLER, (int2)(1, roi_batch_idx));
    int x1          = round(roi_0.y * spatial_scale);
    int y1          = round(roi_0.z * spatial_scale);
    int x2          = round(roi_0.w * spatial_scale);
    int y2          = round(roi_1.x * spatial_scale);
#else
    const int roi_batch_offset = roi_batch_idx * 5;
    FLOAT4 roi_0 = RI_F(roi, SAMPLER, (int2)(0, roi_batch_offset));

    int input_batch = roi_0.x;
    if(input_batch >= in_batch){
        return;
    }
    FLOAT4 roi_1 = RI_F(roi, SAMPLER, (int2)(0, roi_batch_offset + 1));
    FLOAT4 roi_2 = RI_F(roi, SAMPLER, (int2)(0, roi_batch_offset + 2));
    FLOAT4 roi_3 = RI_F(roi, SAMPLER, (int2)(0, roi_batch_offset + 3));
    FLOAT4 roi_4 = RI_F(roi, SAMPLER, (int2)(0, roi_batch_offset + 4));
    int x1          = round(roi_1.x * spatial_scale);
    int y1          = round(roi_2.x * spatial_scale);
    int x2          = round(roi_3.x * spatial_scale);
    int y2          = round(roi_4.x * spatial_scale);
#endif

    int roiW = max(x2 - x1 + 1, 1);
    int roiH = max(y2 - y1 + 1, 1);

    float binSizeW = (float)roiW / (float)out_width;
    float binSizeH = (float)roiH / (float)out_height;

    int hStart = min(max(y1 + (int)floor(out_height_idx * binSizeH), 0), in_height);
    int hEnd   = min(max(y1 + (int)ceil((out_height_idx + 1) * binSizeH), 0), in_height);
    int hLen   = hEnd - hStart;

    int wStart = min(max(x1 + (int)floor(out_width_idx * binSizeW), 0), in_width);
    int wEnd   = min(max(x1 + (int)ceil((out_width_idx + 1) * binSizeW), 0), in_width);
    int wLen   = wEnd - wStart;

    const int pos = mad24(out_channel_idx, out_width, out_width_idx);

    const FLOAT4 zero_vec = (FLOAT4)(0);
    if (wLen <= 0 || hLen <= 0) {
        WI_F(output, (int2)(pos, out_hb_idx), zero_vec);
        return;
    }

    FLOAT4 res = (FLOAT4)(MIN_VALUE);

    const int in_height_start   = hStart;
    const int in_width_start    = wStart;
    const int in_channel_offset = mul24(out_channel_idx, in_width);
    const int in_height_offset = mul24(input_batch, in_height);

    const int batch_idx = mul24(input_batch, in_height);

    for (int height = 0; height < hLen; ++height) {
        int in_height_idx = in_height_start + height;
        for (int width = 0; width < wLen; ++width) {
            int in_width_idx = in_width_start + width;
            FLOAT4 in    = RI_F(input, SAMPLER, (int2)(in_channel_offset + in_width_idx, in_height_offset + in_height_idx));
            res              = fmax(res, in);
        }
    }

    WI_F(output, (int2)(pos, out_hb_idx), res);
}
