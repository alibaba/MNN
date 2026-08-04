#ifndef MNN_METAL_SHADER_AUTO_GENERATE_H
#define MNN_METAL_SHADER_AUTO_GENERATE_H
// With MNN_METAL_PACK_SHADER the shader text is compressed at build time and
// these names become accessor macros supplied by the generated header.
#ifdef MNN_METAL_PACK_SHADER
#include "MetalPackedShader.hpp"
#else
extern const char* shader_MetalReLU6_metal;
extern const char* shader_MetalROIAlign_metal;
extern const char* shader_MetalConvolutionDepthwise_metal;
extern const char* shader_MetalConvolutionActivation_metal;
extern const char* shader_MetalConvolution_metal;
extern const char* shader_MetalSoftmax_metal;
extern const char* shader_MetalLayerNorm_metal;
extern const char* shader_MetalConvolutionWinograd_metal;
extern const char* shader_MetalMatMul_metal;
extern const char* shader_MetalDeconvolution_metal;
extern const char* shader_MetalPooling_metal;
extern const char* shader_MetalROIPooling_metal;
extern const char* shader_MetalConvolution1x1_metal;
extern const char* shader_MetalConvolutionGEMM_metal;
extern const char* shader_MetalResize_metal;
extern const char* shader_MetalPReLU_metal;
extern const char* shader_MetalDefine_metal;
extern const char* shader_MetalEltwise_metal;
#endif /* MNN_METAL_PACK_SHADER */
#endif
