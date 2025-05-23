#include "opencl_source_map.hpp" 
namespace MNN { 
const char* winogradTransformDest2_5_1 = 
"#ifdef MNN_SUPPORT_FP16\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#endif\n"
"__constant sampler_t SAMPLER=CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
"__kernel void winogradTransformDest(__read_only image2d_t uInput,// 0\n"
" __read_only image2d_t uBias,__write_only image2d_t uOutput,\n"
" __private const int unitWidth,// 3\n"
" __private const int unitHeight,__private const int dstWidth,\n"
" __private const int dstHeight,// 6\n"
" __private const int dstChannelC4,__private const int batchOffset) {\n"
" int2 pos=(int2)(get_global_id(0),get_global_id(1)); \n"
" if (pos.x<unitWidth*unitHeight && pos.y<dstChannelC4) {\n"
" int unitWidth_idx=pos.x % unitWidth;\n"
" int unitHeight_idx=pos.x/unitWidth;\n"
" int srcY=pos.y*unitHeight+unitHeight_idx;\n"
" FLOAT4 bias=RI_F(uBias,SAMPLER,(int2)(pos.y,0));\n"
" \n"
" {\n"
" int oyStart=unitHeight_idx*2;\n"
" int oxStart=unitWidth_idx*2;\n"
" FLOAT4 S00=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*0,srcY));\n"
" FLOAT4 S10=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*1,srcY));\n"
" FLOAT4 S20=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*2,srcY));\n"
" FLOAT4 S30=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*3,srcY));\n"
" FLOAT4 S40=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*4,srcY));\n"
" FLOAT4 S50=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*5,srcY));\n"
" FLOAT4 S01=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*6,srcY));\n"
" FLOAT4 S11=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*7,srcY));\n"
" FLOAT4 S21=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*8,srcY));\n"
" FLOAT4 S31=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*9,srcY));\n"
" FLOAT4 S41=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*10,srcY));\n"
" FLOAT4 S51=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*11,srcY));\n"
" FLOAT4 S02=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*12,srcY));\n"
" FLOAT4 S12=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*13,srcY));\n"
" FLOAT4 S22=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*14,srcY));\n"
" FLOAT4 S32=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*15,srcY));\n"
" FLOAT4 S42=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*16,srcY));\n"
" FLOAT4 S52=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*17,srcY));\n"
" FLOAT4 S03=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*18,srcY));\n"
" FLOAT4 S13=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*19,srcY));\n"
" FLOAT4 S23=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*20,srcY));\n"
" FLOAT4 S33=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*21,srcY));\n"
" FLOAT4 S43=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*22,srcY));\n"
" FLOAT4 S53=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*23,srcY));\n"
" FLOAT4 S04=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*24,srcY));\n"
" FLOAT4 S14=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*25,srcY));\n"
" FLOAT4 S24=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*26,srcY));\n"
" FLOAT4 S34=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*27,srcY));\n"
" FLOAT4 S44=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*28,srcY));\n"
" FLOAT4 S54=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*29,srcY));\n"
" FLOAT4 S05=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*30,srcY));\n"
" FLOAT4 S15=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*31,srcY));\n"
" FLOAT4 S25=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*32,srcY));\n"
" FLOAT4 S35=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*33,srcY));\n"
" FLOAT4 S45=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*34,srcY));\n"
" FLOAT4 S55=RI_F(uInput,SAMPLER,(int2)(unitWidth_idx+unitWidth*35,srcY));\n"
" FLOAT4 m00=+S00+S01+S02+S03+S04;\n"
" FLOAT4 m10=+S10+S11+S12+S13+S14;\n"
" FLOAT4 m20=+S20+S21+S22+S23+S24;\n"
" FLOAT4 m30=+S30+S31+S32+S33+S34;\n"
" FLOAT4 m40=+S40+S41+S42+S43+S44;\n"
" FLOAT4 m50=+S50+S51+S52+S53+S54;\n"
" FLOAT4 m01=+S01-S02+(FLOAT)2.0*S03-(FLOAT)2.0*S04+S05;\n"
" FLOAT4 m11=+S11-S12+(FLOAT)2.0*S13-(FLOAT)2.0*S14+S15;\n"
" FLOAT4 m21=+S21-S22+(FLOAT)2.0*S23-(FLOAT)2.0*S24+S25;\n"
" FLOAT4 m31=+S31-S32+(FLOAT)2.0*S33-(FLOAT)2.0*S34+S35;\n"
" FLOAT4 m41=+S41-S42+(FLOAT)2.0*S43-(FLOAT)2.0*S44+S45;\n"
" FLOAT4 m51=+S51-S52+(FLOAT)2.0*S53-(FLOAT)2.0*S54+S55;\n"
" {\n"
" int ox=oxStart+0;\n"
" int oy=oyStart+0;\n"
" if (ox<dstWidth && oy<dstHeight) {\n"
" int imageOx=ox+pos.y*dstWidth;\n"
" int imageOy=oy+batchOffset*dstHeight;\n"
" FLOAT4 res=bias+m00+m10+m20+m30+m40;\n"
"#ifdef RELU\n"
" res=max(res,(FLOAT4)(0));\n"
"#endif\n"
"#ifdef RELU6\n"
" res=clamp(res,(FLOAT4)(0),(FLOAT4)(6));\n"
"#endif\n"
" WI_F(uOutput,(int2)(imageOx,imageOy),res);\n"
" }\n"
" }\n"
" {\n"
" int ox=oxStart+1;\n"
" int oy=oyStart+0;\n"
" if (ox<dstWidth && oy<dstHeight) {\n"
" int imageOx=ox+pos.y*dstWidth;\n"
" int imageOy=oy+batchOffset*dstHeight;\n"
" FLOAT4 res=bias+m10-m20+(FLOAT)2.0*m30-(FLOAT)2.0*m40+m50;\n"
"#ifdef RELU\n"
" res=max(res,(FLOAT4)(0));\n"
"#endif\n"
"#ifdef RELU6\n"
" res=clamp(res,(FLOAT4)(0),(FLOAT4)(6));\n"
"#endif\n"
" WI_F(uOutput,(int2)(imageOx,imageOy),res);\n"
" }\n"
" }\n"
" {\n"
" int ox=oxStart+0;\n"
" int oy=oyStart+1;\n"
" if (ox<dstWidth && oy<dstHeight) {\n"
" int imageOx=ox+pos.y*dstWidth;\n"
" int imageOy=oy+batchOffset*dstHeight;\n"
" FLOAT4 res=bias+m01+m11+m21+m31+m41;\n"
"#ifdef RELU\n"
" res=max(res,(FLOAT4)(0));\n"
"#endif\n"
"#ifdef RELU6\n"
" res=clamp(res,(FLOAT4)(0),(FLOAT4)(6));\n"
"#endif\n"
" WI_F(uOutput,(int2)(imageOx,imageOy),res);\n"
" }\n"
" }\n"
" {\n"
" int ox=oxStart+1;\n"
" int oy=oyStart+1;\n"
" if (ox<dstWidth && oy<dstHeight) {\n"
" int imageOx=ox+pos.y*dstWidth;\n"
" int imageOy=oy+batchOffset*dstHeight;\n"
" FLOAT4 res=bias+m11-m21+(FLOAT4)2.0*m31-(FLOAT4)2.0*m41+m51;\n"
"#ifdef RELU\n"
" res=max(res,(FLOAT4)(0));\n"
"#endif\n"
"#ifdef RELU6\n"
" res=clamp(res,(FLOAT4)(0),(FLOAT4)(6));\n"
"#endif\n"
" WI_F(uOutput,(int2)(imageOx,imageOy),res);\n"
" }\n"
" }\n"
" }\n"
" }\n"
"}\n"
;
}
