#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void blit( 
                    __read_only image2d_t input, 
                    __write_only image2d_t output,
                    int4 inputOffset,
                    int4 outputOffset,
                    int4 region,
                    int2 inputWH,
                    int2 outputWH,
                    int2 wh
                    ) {
    int w = wh.x;
    int h = wh.y;
    int2 xy = (int2)(get_global_id(0), get_global_id(1));

    //N, C, H, W
    int4 pos = (int4)(xy.y/h, xy.x/w, xy.y%h, xy.x%w);

    if (pos.x < region.x && pos.y < region.y) {
        int4 posInput = inputOffset + pos;
        int4 posOutput = outputOffset + pos;

        int2 outputPos = (int2)(posOutput.w + posOutput.y*outputWH.x, posOutput.x*outputWH.y + posOutput.z);
        int2 inputPos = (int2)(posInput.w + posInput.y*inputWH.x, posInput.x*inputWH.y + posInput.z);

        WI_F(output, outputPos, RI_F(input, SAMPLER, inputPos));
    }
}

