#include "opencl_source_map.hpp" 
namespace MNN { 
#ifndef MNN_OPENCL_BUFFER_CLOSED
const char* input_transe_buf = 
"#ifdef MNN_SUPPORT_FP16\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#endif\n"
"__attribute__((intel_reqd_sub_group_size(16)))\n"
"__kernel void conv_transe_c4_c1(\n"
" int global_size_dim0,\n"
" int global_size_dim1,\n"
" int global_size_dim2,\n"
" __global FLOAT* input,\n"
" __global FLOAT* output,\n"
" __private const int input_width,\n"
" __private const int input_height,\n"
" __private const int input_channel,\n"
" __private const int batch,\n"
" __private const int channel_blocks,\n"
" __private const int input_pad_left,\n"
" __private const int input_pad_right)\n"
"{\n"
" int x=get_global_id(0);\n"
" int w=x % input_width;\n"
" int h=x/input_width;\n"
" int c=get_global_id(1);\n"
" int b=get_global_id(2);\n"
" int cout=c << 2;\n"
" if(x >= global_size_dim0 || c >= global_size_dim1 || b >= global_size_dim2)\n"
" return;\n"
" // Input offset calculations:\n"
" const uint input_x_pitch=4;\n"
" const uint input_y_pitch=input_x_pitch*input_width;\n"
" const uint input_f_pitch=input_y_pitch*input_height;\n"
" const uint input_b_pitch=input_f_pitch*batch;\n"
" const uint input_offset=b*input_f_pitch +\n"
" c*input_b_pitch +\n"
" h*input_y_pitch +\n"
" w*input_x_pitch;\n"
" // Output offset calculations:\n"
" const uint output_x_pitch=1;\n"
" const uint output_y_pitch=output_x_pitch*input_width;\n"
" const uint output_f_pitch=output_y_pitch*input_height;\n"
" const uint output_b_pitch=output_f_pitch*input_channel;\n"
" const uint output_offset=b*output_b_pitch +\n"
" cout*output_f_pitch+\n"
" h*output_y_pitch +\n"
" w*output_x_pitch;\n"
" \n"
" FLOAT4 value=vload4(0,input+input_offset);\n"
" FLOAT *value_ptr=(FLOAT*)&value;\n"
" for(int i=0; i<4 && cout+i<input_channel; ++i){\n"
" output[output_offset+i*output_f_pitch]=value_ptr[i];\n"
" }\n"
"}\n"
"__attribute__((intel_reqd_sub_group_size(16)))\n"
"__kernel void conv_transe_c4_c16(\n"
" int global_size_dim0,\n"
" int global_size_dim1,\n"
" int global_size_dim2,\n"
" __global FLOAT* input,\n"
" __global FLOAT* output,\n"
" int input_width,\n"
" int input_height,\n"
" int input_channel,\n"
" int batch,\n"
" int channel_blocks,\n"
" int input_pad_left,\n"
" int input_pad_right)\n"
"{\n"
" int x=get_global_id(0);\n"
" int w=x % input_width;\n"
" int h=x/input_width;\n"
" int c=get_global_id(1);\n"
" int b=get_global_id(2);\n"
" int cout=c >> 2;\n"
" if(x >= global_size_dim0 || c >= global_size_dim1 || b >= global_size_dim2)\n"
" return;\n"
" \n"
" // Input offset calculations:\n"
" const uint input_x_pitch=4;\n"
" const uint input_y_pitch=input_x_pitch*input_width;\n"
" const uint input_f_pitch=input_y_pitch*input_height;\n"
" const uint input_b_pitch=input_f_pitch*batch;\n"
" \n"
" const uint input_offset=b*input_f_pitch +\n"
" c*input_b_pitch +\n"
" h*input_y_pitch +\n"
" w*input_x_pitch;\n"
" \n"
" // Output offset calculations:\n"
" const uint output_x_pitch=16;\n"
" const uint output_y_pitch=output_x_pitch*(input_pad_left+input_width+input_pad_right);\n"
" const uint output_f_pitch=output_y_pitch*input_height;\n"
" const uint output_b_pitch=output_f_pitch*((input_channel+15)/16);\n"
" \n"
" const uint output_offset=b*output_b_pitch +\n"
" cout*output_f_pitch+\n"
" h*output_y_pitch +\n"
" (w+input_pad_left)*output_x_pitch+(c % 4)*4;\n"
" \n"
" FLOAT4 value=vload4(0,input+input_offset);\n"
" vstore4(value,0,output+output_offset);\n"
" if(w == 0){\n"
" uint pad_offset=b*output_b_pitch+cout*output_f_pitch+h*output_y_pitch+(c % 4)*4;\n"
" for(int i=0; i<input_pad_left; ++i){\n"
" vstore4((FLOAT4)0,0,output+pad_offset+i*output_x_pitch);\n"
" }\n"
" pad_offset += (input_pad_left+input_width)*output_x_pitch;\n"
" for(int i=0; i<input_pad_right; ++i){\n"
" vstore4((FLOAT4)0,0,output+pad_offset+i*output_x_pitch);\n"
" }\n"
" }\n"
"}\n"
;
#endif
}
