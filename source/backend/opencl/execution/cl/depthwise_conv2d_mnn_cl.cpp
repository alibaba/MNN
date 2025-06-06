#include "opencl_source_map.hpp" 
namespace MNN { 
const char* depthwise_conv2d = 
"#ifdef MNN_SUPPORT_FP16\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#endif\n"
"#define READ_INPUT_IMAGE(i, base) "" int inOffset##i = inWidthOffset##i + base; "" inOffset##i = "" select(inCurIdx + inOffset##i, -1, (inOffset##i < 0 || inOffset##i >= inputShape.y)); "" inValue##i=RI_F(input,SAMPLER,(int2)(inOffset##i,inHeightIdx));\n"
"#define CALCULATE_OUTPUT(i) "" outValue##i = mad(inValue##i.x, weights0, outValue##i); "" outValue##i = mad(inValue##i.y, weights1, outValue##i); "" outValue##i = mad(inValue##i.z, weights2, outValue##i); "" outValue##i=mad(inValue##i.w,weights3,outValue##i);\n"
"#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0,__private const int global_size_dim1,\n"
"__constant sampler_t SAMPLER=CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
"#define DEAL_NON_UNIFORM_DIM2(input1, input2) "" if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { "" return; "" }\n"
"__kernel\n"
"#ifdef SET_ATTRIBUTE\n"
"__attribute__((work_group_size_hint(16,16,1)))\n"
"#endif\n"
"void depthwise_conv2d_s1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,__read_only image2d_t filter,\n"
" #ifndef NO_BIAS\n"
" __read_only image2d_t bias,\n"
" #endif\n"
" __write_only image2d_t output,\n"
" __private const int2 inputShape,\n"
" __private const int inChannelBlocks,\n"
" __private const int2 outputShape,\n"
" __private const int2 filterShape,\n"
" __private const int2 paddingShape) {\n"
" const int outChannelWidthIdx=get_global_id(0);\n"
" const int outHeightBlockIdx=get_global_id(1);\n"
" DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx,outHeightBlockIdx);\n"
" int ow4=(outputShape.y+3)/4;\n"
" const int outChannelBlockIdx=outChannelWidthIdx/ow4;\n"
" const int outWidthBlockidx=outChannelWidthIdx % ow4;\n"
" const int inChannelBlockIdx=outChannelBlockIdx;\n"
" #ifndef NO_BIAS\n"
" FLOAT4 outValue0=RI_F(bias,SAMPLER,(int2)(outChannelBlockIdx,0));\n"
" #else\n"
" FLOAT4 outValue0=(FLOAT4)(0.0f);\n"
" #endif\n"
" FLOAT4 outValue1=outValue0;\n"
" FLOAT4 outValue2=outValue0;\n"
" FLOAT4 outValue3=outValue0;\n"
" const int outWidthBlockidx4=outWidthBlockidx << 2;\n"
" const int inWidthOffset0=outWidthBlockidx4-paddingShape.y;\n"
" const int inWidthOffset1=inWidthOffset0+1;\n"
" const int inWidthOffset2=inWidthOffset0+2;\n"
" const int inWidthOffset3=inWidthOffset0+3;\n"
" int heightIdx=outHeightBlockIdx % outputShape.x-paddingShape.x;\n"
" const int outBatchIdx=mul24((outHeightBlockIdx/outputShape.x),inputShape.x);\n"
" const int inCurIdx=mul24(inChannelBlockIdx,inputShape.y);\n"
" const int inWidthIdx0=select(inCurIdx+inWidthOffset0,-1,(inWidthOffset0<0 || inWidthOffset0 >= inputShape.y));\n"
" const int inWidthIdx1=select(inCurIdx+inWidthOffset1,-1,(inWidthOffset1<0 || inWidthOffset1 >= inputShape.y));\n"
" const int inWidthIdx2=select(inCurIdx+inWidthOffset2,-1,(inWidthOffset2<0 || inWidthOffset2 >= inputShape.y));\n"
" FLOAT4 inValue0,inValue1,inValue2,inValue3;\n"
" for (int kh=0; kh<filterShape.x; kh++) {\n"
" int inHeightIdx=select(heightIdx+outBatchIdx,-1,(heightIdx<0 || heightIdx >= inputShape.x));\n"
" heightIdx++;\n"
" inValue1=RI_F(input,SAMPLER,(int2)(inWidthIdx0,inHeightIdx));\n"
" inValue2=RI_F(input,SAMPLER,(int2)(inWidthIdx1,inHeightIdx));\n"
" inValue3=RI_F(input,SAMPLER,(int2)(inWidthIdx2,inHeightIdx));\n"
" for (int kw=0; kw<filterShape.y; kw++) {\n"
" int filterIdx=mad24(kh,filterShape.y,kw);\n"
" inValue0=inValue1;\n"
" inValue1=inValue2;\n"
" inValue2=inValue3;\n"
" int inWidthIdx=inWidthOffset3+kw;\n"
" inWidthIdx=select(inCurIdx+inWidthIdx,-1,(inWidthIdx<0 || inWidthIdx >= inputShape.y));\n"
" inValue3=RI_F(input,SAMPLER,(int2)(inWidthIdx,inHeightIdx));\n"
" FLOAT4 weights=RI_F(filter,SAMPLER,(int2)(filterIdx,inChannelBlockIdx));\n"
" outValue0=mad(inValue0,weights,outValue0);\n"
" outValue1=mad(inValue1,weights,outValue1);\n"
" outValue2=mad(inValue2,weights,outValue2);\n"
" outValue3=mad(inValue3,weights,outValue3);\n"
" }\n"
" }\n"
"#ifdef RELU\n"
" outValue0=fmax(outValue0,(FLOAT4)0);\n"
" outValue1=fmax(outValue1,(FLOAT4)0);\n"
" outValue2=fmax(outValue2,(FLOAT4)0);\n"
" outValue3=fmax(outValue3,(FLOAT4)0);\n"
"#endif\n"
"#ifdef RELU6\n"
" outValue0=clamp(outValue0,(FLOAT4)0,(FLOAT4)6);\n"
" outValue1=clamp(outValue1,(FLOAT4)0,(FLOAT4)6);\n"
" outValue2=clamp(outValue2,(FLOAT4)0,(FLOAT4)6);\n"
" outValue3=clamp(outValue3,(FLOAT4)0,(FLOAT4)6);\n"
"#endif\n"
" const int remain=outputShape.y-outWidthBlockidx4;\n"
" int outWidthIdx=mul24(outChannelBlockIdx,outputShape.y)+outWidthBlockidx4;\n"
" if (remain >= 4) {\n"
" WI_F(output,(int2)(outWidthIdx,outHeightBlockIdx),outValue0);\n"
" WI_F(output,(int2)(outWidthIdx+1,outHeightBlockIdx),outValue1);\n"
" WI_F(output,(int2)(outWidthIdx+2,outHeightBlockIdx),outValue2);\n"
" WI_F(output,(int2)(outWidthIdx+3,outHeightBlockIdx),outValue3);\n"
" } else if (remain == 3) {\n"
" WI_F(output,(int2)(outWidthIdx,outHeightBlockIdx),outValue0);\n"
" WI_F(output,(int2)(outWidthIdx+1,outHeightBlockIdx),outValue1);\n"
" WI_F(output,(int2)(outWidthIdx+2,outHeightBlockIdx),outValue2);\n"
" } else if (remain == 2) {\n"
" WI_F(output,(int2)(outWidthIdx,outHeightBlockIdx),outValue0);\n"
" WI_F(output,(int2)(outWidthIdx+1,outHeightBlockIdx),outValue1);\n"
" } else if (remain == 1) {\n"
" WI_F(output,(int2)(outWidthIdx,outHeightBlockIdx),outValue0);\n"
" }\n"
"}\n"
"__kernel\n"
"#ifdef SET_ATTRIBUTE\n"
"__attribute__((work_group_size_hint(16,16,1)))\n"
"#endif\n"
"void depthwise_conv2d(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,__read_only image2d_t filter,\n"
" #ifndef NO_BIAS\n"
" __read_only image2d_t bias,\n"
" #endif\n"
" __write_only image2d_t output,\n"
" __private const int2 inputShape,\n"
" __private const int inChannelBlocks,__private const int2 outputShape,\n"
" __private const int2 filterShape,\n"
" __private const int2 paddingShape,\n"
" __private const int2 dilationShape,\n"
" __private const int2 strideShape) {\n"
" const int outChannelWidthIdx=get_global_id(0);\n"
" const int outHeightIdx=get_global_id(1);\n"
" DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx,outHeightIdx);\n"
" int ow4=(outputShape.y+3)/4;\n"
" const int outChannelBlockIdx=outChannelWidthIdx/ow4;\n"
" const int outWidthBlockidx=outChannelWidthIdx % ow4;\n"
" const int inChannelBlockIdx=outChannelBlockIdx;\n"
" #ifndef NO_BIAS\n"
" FLOAT4 outValue0=RI_F(bias,SAMPLER,(int2)(outChannelBlockIdx,0));\n"
" #else\n"
" FLOAT4 outValue0=(FLOAT4)(0.0f);\n"
" #endif\n"
" FLOAT4 outValue1=outValue0;\n"
" FLOAT4 outValue2=outValue0;\n"
" FLOAT4 outValue3=outValue0;\n"
" const int inWidthOffset0=mad24(outWidthBlockidx,strideShape.y << 2,-paddingShape.y);\n"
" const int inWidthOffset1=inWidthOffset0+strideShape.y;\n"
" const int inWidthOffset2=inWidthOffset1+strideShape.y;\n"
" const int inWidthOffset3=inWidthOffset2+strideShape.y;\n"
" int heightIdx=mad24(outHeightIdx % outputShape.x,strideShape.x,-paddingShape.x);\n"
" const int outBatchIdx=mul24((outHeightIdx/outputShape.x),inputShape.x);\n"
" const int inCurIdx=mul24(inChannelBlockIdx,inputShape.y);\n"
" for (int kh=0; kh<filterShape.x; kh++) {\n"
" int inHeightIdx=select(heightIdx+outBatchIdx,-1,(heightIdx<0 || heightIdx >= inputShape.x));\n"
" heightIdx += dilationShape.x;\n"
" for (int kw=0; kw<filterShape.y; kw++) {\n"
" int filterIdx=mad24(kh,filterShape.y,kw);\n"
" FLOAT4 inValue0,inValue1,inValue2,inValue3;\n"
" int inWidthIdx=mul24(kw,dilationShape.y);\n"
" READ_INPUT_IMAGE(0,inWidthIdx);\n"
" READ_INPUT_IMAGE(1,inWidthIdx);\n"
" READ_INPUT_IMAGE(2,inWidthIdx);\n"
" READ_INPUT_IMAGE(3,inWidthIdx);\n"
" FLOAT4 weights=RI_F(filter,SAMPLER,(int2)(filterIdx,inChannelBlockIdx));\n"
" outValue0=mad(inValue0,weights,outValue0);\n"
" outValue1=mad(inValue1,weights,outValue1);\n"
" outValue2=mad(inValue2,weights,outValue2);\n"
" outValue3=mad(inValue3,weights,outValue3);\n"
" }\n"
" }\n"
"#ifdef RELU\n"
" outValue0=fmax(outValue0,(FLOAT4)0);\n"
" outValue1=fmax(outValue1,(FLOAT4)0);\n"
" outValue2=fmax(outValue2,(FLOAT4)0);\n"
" outValue3=fmax(outValue3,(FLOAT4)0);\n"
"#endif\n"
"#ifdef RELU6\n"
" outValue0=clamp(outValue0,(FLOAT4)0,(FLOAT4)6);\n"
" outValue1=clamp(outValue1,(FLOAT4)0,(FLOAT4)6);\n"
" outValue2=clamp(outValue2,(FLOAT4)0,(FLOAT4)6);\n"
" outValue3=clamp(outValue3,(FLOAT4)0,(FLOAT4)6);\n"
"#endif\n"
" const int outWidthBlockidx4=outWidthBlockidx << 2;\n"
" const int remain=outputShape.y-outWidthBlockidx4;\n"
" int outWidthIdx=mul24(outChannelBlockIdx,outputShape.y)+outWidthBlockidx4;\n"
" if (remain >= 4) {\n"
" WI_F(output,(int2)(outWidthIdx,outHeightIdx),outValue0);\n"
" WI_F(output,(int2)(outWidthIdx+1,outHeightIdx),outValue1);\n"
" WI_F(output,(int2)(outWidthIdx+2,outHeightIdx),outValue2);\n"
" WI_F(output,(int2)(outWidthIdx+3,outHeightIdx),outValue3);\n"
" } else if (remain == 3) {\n"
" WI_F(output,(int2)(outWidthIdx,outHeightIdx),outValue0);\n"
" WI_F(output,(int2)(outWidthIdx+1,outHeightIdx),outValue1);\n"
" WI_F(output,(int2)(outWidthIdx+2,outHeightIdx),outValue2);\n"
" } else if (remain == 2) {\n"
" WI_F(output,(int2)(outWidthIdx,outHeightIdx),outValue0);\n"
" WI_F(output,(int2)(outWidthIdx+1,outHeightIdx),outValue1);\n"
" } else if (remain == 1) {\n"
" WI_F(output,(int2)(outWidthIdx,outHeightIdx),outValue0);\n"
" }\n"
"}\n"
;
}
