#include "ShaderMap.hpp"
#include "AllShader.hpp"
namespace MNN {
void ShaderMap::init() {
mMaps.insert(std::make_pair("shader_MetalReLU6_metal", shader_MetalReLU6_metal));
mMaps.insert(std::make_pair("shader_MetalConvolutionDepthwise_metal", shader_MetalConvolutionDepthwise_metal));
mMaps.insert(std::make_pair("shader_MetalConvolutionActivation_metal", shader_MetalConvolutionActivation_metal));
mMaps.insert(std::make_pair("shader_MetalConvolution_metal", shader_MetalConvolution_metal));
mMaps.insert(std::make_pair("shader_MetalReduction_metal", shader_MetalReduction_metal));
mMaps.insert(std::make_pair("shader_MetalSoftmax_metal", shader_MetalSoftmax_metal));
mMaps.insert(std::make_pair("shader_MetalLayerNorm_metal", shader_MetalLayerNorm_metal));
mMaps.insert(std::make_pair("shader_MetalConvolutionWinograd_metal", shader_MetalConvolutionWinograd_metal));
mMaps.insert(std::make_pair("shader_MetalMatMul_metal", shader_MetalMatMul_metal));
mMaps.insert(std::make_pair("shader_MetalScale_metal", shader_MetalScale_metal));
mMaps.insert(std::make_pair("shader_MetalDeconvolution_metal", shader_MetalDeconvolution_metal));
mMaps.insert(std::make_pair("shader_MetalPooling_metal", shader_MetalPooling_metal));
mMaps.insert(std::make_pair("shader_MetalROIPooling_metal", shader_MetalROIPooling_metal));
mMaps.insert(std::make_pair("shader_MetalConvolution1x1_metal", shader_MetalConvolution1x1_metal));
mMaps.insert(std::make_pair("shader_MetalConvolutionGEMM_metal", shader_MetalConvolutionGEMM_metal));
mMaps.insert(std::make_pair("shader_MetalResize_metal", shader_MetalResize_metal));
mMaps.insert(std::make_pair("shader_MetalPReLU_metal", shader_MetalPReLU_metal));
mMaps.insert(std::make_pair("shader_MetalDefine_metal", shader_MetalDefine_metal));
mMaps.insert(std::make_pair("shader_MetalEltwise_metal", shader_MetalEltwise_metal));
}
}
