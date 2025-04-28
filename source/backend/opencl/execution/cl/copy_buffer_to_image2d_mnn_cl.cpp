#include "opencl_source_map.hpp" 
namespace MNN { 
const char* copy_buffer_to_image2d = 
"#ifdef MNN_SUPPORT_FP16\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#endif\n"
"__constant sampler_t SAMPLER=CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
"__kernel void copy_buffer_to_image2d(\n"
" #ifdef BUFFER_INP_FP32\n"
" __global const float4* input,\n"
" #else\n"
" __global const FLOAT4* input,\n"
" #endif\n"
" __write_only image2d_t uOutput,\n"
" __private const int width,__private const int height) {\n"
" int x=get_global_id(0);\n"
" int y=get_global_id(1);\n"
" if (x<width && y<height) {\n"
" WI_F(uOutput,(int2)(x,y),(FLOAT4)((FLOAT)input[x+y*width].x,(FLOAT)input[x+y*width].y,(FLOAT)input[x+y*width].z,(FLOAT)input[x+y*width].w));\n"
" }\n"
"}\n"
;
}
