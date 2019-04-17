#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void blitBuffer(
                    const __global FLOAT *input, 
                    __global FLOAT *output,
                    int4 inputOffset,
                    int4 outputOffset,
                    int4 region,
                    int4 inputStride,
                    int4 outputStride,
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

        int outputPos = posOutput.x * outputStride.x
            + posOutput.y * outputStride.y
            + posOutput.z * outputStride.z
            + posOutput.w * outputStride.w;
        
        int inputPos = posInput.x * inputStride.x
            + posInput.y * inputStride.y
            + posInput.z * inputStride.z
            + posInput.w * inputStride.w;
        
        output[outputPos] = input[inputPos];
    }
}

__kernel void blitImageToBuffer(
                    __read_only image2d_t input, 
                    __global FLOAT *output,
                    int4 inputOffset,
                    int4 outputOffset,
                    int4 region,
                    int2 inputWH,
                    int4 outputStride,
                    int4 outputSize/*nhwc*/
                    ) {
    int w = outputSize.z;
    int h = outputSize.y;
    int c = outputSize.w;
    int n = outputSize.x;
    int2 xy = (int2)(get_global_id(0), get_global_id(1));

    //N, C, H, W
    int4 pos = (int4)(xy.y/h, xy.x/w, xy.y%h, xy.x%w);
    int4 bufferPos = pos * (int4)(1, 4, 1, 1);

    if (pos.x < region.x && pos.y < region.y) {
        int4 posInput = inputOffset + pos;
        int4 posOutput = outputOffset + bufferPos;
        int2 inputPos = (int2)(posInput.w + posInput.y*inputWH.x, posInput.x*inputWH.y + posInput.z);

        FLOAT4 color = RI_F(input, SAMPLER, inputPos);

        int outputPosBasic = posOutput.x*outputStride.x
                + posOutput.y*outputStride.y
                + posOutput.z*outputStride.z
                + posOutput.w*outputStride.w;

        int outputPos0 = outputPosBasic + 0*outputStride.y;
        output[outputPos0] = color.x;
        if (posOutput.y + 1 < c) {
            int outputPos1 = outputPosBasic + 1*outputStride.y;
            output[outputPos1] = color.y;
        }
        if (posOutput.y + 2 < c) {
            int outputPos1 = outputPosBasic + 2*outputStride.y;
            output[outputPos1] = color.z;
        }
        if (posOutput.y + 3 < c) {
            int outputPos1 = outputPosBasic + 3*outputStride.y;
            output[outputPos1] = color.w;
        }
    }
}

__kernel void blitBufferToImage(
                    __global FLOAT *input,
                    __write_only image2d_t output,
                    int4 inputOffset,
                    int4 outputOffset,
                    int4 region,
                    int4 inputStride,
                    int2 outputWH,
                    int2 wh
                    ) {
    int w = wh.x;
    int h = wh.y;
    int2 xy = (int2)(get_global_id(0), get_global_id(1));

    //N, C, H, W
    int4 pos = (int4)(xy.y/h, xy.x/w, xy.y%h, xy.x%w);
    int4 bufferPos = pos * (int4)(1, 4, 1, 1);

    if (pos.x < region.x && pos.y < region.y) {
        int4 posInput = inputOffset + bufferPos;
        int4 posOutput = outputOffset + pos;
        int2 outputPos = (int2)(posOutput.w + posOutput.y*outputWH.x, posOutput.x*outputWH.y + posOutput.z);
        int inputPosBasic = posInput.x*inputStride.x 
                +posInput.y*inputStride.y
                +posInput.z*inputStride.z
                +posInput.w*inputStride.w;

        int inputPos0 = inputPosBasic + 0*inputStride.y;
        int inputPos1 = inputPosBasic + 1*inputStride.y;
        int inputPos2 = inputPosBasic + 2*inputStride.y;
        int inputPos3 = inputPosBasic + 3*inputStride.y;

        FLOAT4 color;
        color.x = input[inputPos0];
        color.y = input[inputPos1];
        color.z = input[inputPos2];
        color.w = input[inputPos3];

        WI_F(output, outputPos, color);
    }
}

