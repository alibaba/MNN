#include "opencl_source_map.hpp" 
namespace MNN { 
const char* range = 
"#ifdef MNN_SUPPORT_FP16\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#endif\n"
"#define GLOBAL_SIZE_3_DIMS ""__private const int global_size_dim0,__private const int global_size_dim1,__private const int global_size_dim2,\n"
"#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3) "" if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { "" return; "" }\n"
"__constant sampler_t SAMPLER=CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
"__kernel void range(GLOBAL_SIZE_3_DIMS\n"
" __read_only image2d_t input0,\n"
" __read_only image2d_t input2,\n"
" __write_only image2d_t output,\n"
" __private const int width,\n"
" __private const int height,\n"
" __private const int channel,\n"
" __private const int channelBlock\n"
" ) {\n"
" const int width_idx=get_global_id(0);\n"
" const int height_idx=get_global_id(1);\n"
" const int batch_channel_idx=get_global_id(2);\n"
" DEAL_NON_UNIFORM_DIM3(width_idx,height_idx,batch_channel_idx);\n"
" \n"
" const int batch_idx=batch_channel_idx/channelBlock;\n"
" const int channel_idx=batch_channel_idx % channelBlock;\n"
" \n"
" const int bh=batch_idx*height+height_idx;\n"
" const int cw=channel_idx*width+width_idx;\n"
" const int channel4=channel_idx << 2;\n"
" int index=(((batch_idx*channel)+channel4)*height+height_idx)*width+width_idx;\n"
" int size=height*width;\n"
" int4 index4=(int4)(index,index+size,index+size*2,index+size*3);\n"
" INPUT_TYPE_I start=RI_DATA(input0,SAMPLER,(int2)(0,0)).x;\n"
" INPUT_TYPE_I step=RI_DATA(input2,SAMPLER,(int2)(0,0)).x;\n"
" OUTPUT_TYPE_I4 value=(OUTPUT_TYPE_I4)start+CONVERT_OUTPUT_I4(index4)*(OUTPUT_TYPE_I4)step;\n"
" WI_DATA(output,(int2)(cw,bh),value);\n"
"}\n"
;
}
