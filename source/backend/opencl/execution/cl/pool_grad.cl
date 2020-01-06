#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void maxpool_grad(__read_only image2d_t originInput, __read_only image2d_t originOutput, __read_only image2d_t inputGrad, __write_only image2d_t output, int2 shape, int2 poolShape, int2 kernelSize, int2 stride) {
    const int2 pos = (int2)(get_global_id(1), get_global_id(0)); // read_imagef and write_imagef need (w, h) layout position
    const int h = pos.y % shape.x, w = pos.x % shape.y;
    const int hOffset_ = pos.y - h, wOffset_ = pos.x - w;
    const int hOffset = (pos.y / shape.x) * poolShape.x, wOffset = (pos.x / shape.y) * poolShape.y;
    const int hStart = ceil((float)(h - kernelSize.x + 1) / stride.x), hEnd = floor((float)h / stride.x);
    const int wStart = ceil((float)(w - kernelSize.y + 1) / stride.y), wEnd = floor((float)w / stride.y);
    FLOAT4 in0 = RI_F(originInput, SAMPLER, pos), res = 0;
    for (int i = hStart; i <= hEnd; ++i) {
        for (int j = wStart; j <= wEnd; ++j) {
            FLOAT4 in1 = RI_F(originOutput, SAMPLER, (int2)(wOffset + j, hOffset + i));
            if (!any(isequal(in0, in1))) {
                continue;
            }
            FLOAT4 grad = RI_F(inputGrad, SAMPLER, (int2)(wOffset + j, hOffset + i));
            FLOAT4 flag = 1;
            const int hStart_ = i * stride.x;
            const int wStart_ = j * stride.y, wEnd_ = wStart_ + kernelSize.y;
            for (int i_ = hStart_; i_ < h; ++i_) {
                for (int j_ = wStart_; j_ < wEnd_; ++j_) {
                    FLOAT4 in0_ = RI_F(originInput, SAMPLER, (int2)(wOffset_ + j_, hOffset_ + i_));
                    flag = flag * select((FLOAT4)1, (FLOAT4)0, isequal(in1, in0_));
                }
            }
            for (int j_ = wStart_; j_ < w; ++j_) {
                FLOAT4 in0_ = RI_F(originInput, SAMPLER, (int2)(wOffset_ + j_, hOffset_ + h));
                flag = flag * select((FLOAT4)1, (FLOAT4)0, isequal(in1, in0_));
            }
            res = res + select((FLOAT4)0, grad * flag, isequal(in0, in1));
        }
    }
    WI_F(output, pos, res);
}

__kernel void avepool_grad(__read_only image2d_t inputGrad, __write_only image2d_t output, int2 shape, int2 poolShape, int2 kernelSize, int2 stride) {
    const int2 pos = (int2)(get_global_id(1), get_global_id(0));
    const int h = pos.y % shape.x, w = pos.x % shape.y;
    const int hOffset = (pos.y / shape.x) * poolShape.x, wOffset = (pos.x / shape.y) * poolShape.y;
    const int hStart = ceil((float)(h - kernelSize.x + 1) / stride.x), hEnd = floor((float)h / stride.x);
    const int wStart = ceil((float)(w - kernelSize.y + 1) / stride.y), wEnd = floor((float)w / stride.y);
    FLOAT4 sum = 0;
    for (int i = hStart; i <= hEnd; ++i) {
        for (int j = wStart; j <= wEnd; ++j) {
            sum = sum + RI_F(inputGrad, SAMPLER, (int2)(wOffset + j, hOffset + i));
        }
    }
    WI_F(output, pos, sum / (kernelSize.x * kernelSize.y));
}
