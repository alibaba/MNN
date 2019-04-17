#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void softmax_height(__read_only image2d_t input, __write_only image2d_t output, 
                      __private const int4 shape // NCHW
                      ) {
    int wc = get_global_id(0);
    int b = get_global_id(1);
    if (wc < shape.y*shape.w && b < shape.x) {
        /*Compute Max */
        FLOAT4 maxValue = RI_F(input, (int2)(wc, b*shape.z));
        for (int i=1; i<shape.z; ++i) {
            maxValue = fmax(maxValue, RI_F(input, SAMPLER, (int2)(wc, b*shape.z+i)));
        }
        /*Compute Exp Sum*/
        FLOAT4 sumValue = (FLOAT4)0;
        for (int i=0; i<shape.z; ++i) {
            sumValue += exp(RI_F(input, SAMPLER, (int2)(wc, b*shape.z+i)) - maxValue);
        }
        /*Compute Result */
        for (int i=0; i<shape.z; ++i) {
            FLOAT4 value = exp(RI_F(input, SAMPLER, (int2)(wc, b*shape.z+i)) - maxValue) / sumValue;
            WI_F(output, (int2)(wc, b*shape.z+i), value);
        }
    }    
}
