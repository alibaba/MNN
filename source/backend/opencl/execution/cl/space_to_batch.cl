#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void space_to_batch(__read_only image2d_t uInput, __write_only image2d_t uOutput,
                             __private const int4 inImageSize, __private const int4 outImgSize,
                             __private const int2 padding, __private const int2 blockShape) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < outImgSize.x && pos.y < outImgSize.y) {
        // pos.x -> w, pos.y -> h, pos.z -> c4 * b;
        int outBatchIndex    = pos.z / outImgSize.z;
        int outChannelIndex  = pos.z % outImgSize.z;
        int inBatchIndex     = outBatchIndex % inImageSize.w;
        int sw               = (outBatchIndex / inImageSize.w) % blockShape.y;
        int sh               = (outBatchIndex / inImageSize.w) / blockShape.y;
        int validHeightStart = max(0, ((padding.x - sh + blockShape.x - 1) / blockShape.x));
        int validHeightEnd   = min(outImgSize.y, ((inImageSize.y + padding.x - sh + blockShape.x - 1) / blockShape.x));
        int validWidthStart  = max(0, ((padding.y - sw + blockShape.y - 1) / blockShape.y));
        int validWidthEnd    = min(outImgSize.x, ((inImageSize.x + padding.y - sw + blockShape.y - 1) / blockShape.y));

        int inPosX = pos.x * blockShape.y + sw - padding.y;
        int inPosY = pos.y * blockShape.x + sh - padding.x;
        int inPosZ = inBatchIndex * inImageSize.z + outChannelIndex;

        int inputX = select(inPosX + inPosZ * inImageSize.x, -1, pos.x < validWidthStart || pos.x >= validWidthEnd);
        int inputY =
            select(inPosY + inBatchIndex * inImageSize.y, -1, pos.y < validHeightStart || pos.y >= validHeightEnd);

        FLOAT4 res = RI_F(uInput, SAMPLER, (int2)(inputX, inputY));
        WI_F(uOutput, (int2)(pos.x + outChannelIndex * outImgSize.x, pos.y + outBatchIndex * outImgSize.y), res);
    }
}
