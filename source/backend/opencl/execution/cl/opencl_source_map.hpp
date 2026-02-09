#include <map> 
#include <string> 
#include <vector> 
namespace MNN { 
extern const char* conv_2d;
extern const char* deconv_2d;
extern const char* unary;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* grid_sample_buf;
#endif
extern const char* interp;
extern const char* select;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* range_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* self_attention_buf;
#endif
extern const char* winogradTransformSource2_3_1;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* gemv_conv1x1_buf;
#endif
extern const char* raster;
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* conv_2d_c1_subgroup_buf;
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* linear_attention_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* matmul_local_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* conv_2d_int_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* interp_buf;
#endif
extern const char* scale;
extern const char* softmax;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* binary_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* raster_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* binary_subgroup_buf;
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* depthwise_conv2d_subgroup_buf;
#endif
#endif
extern const char* nearest;
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* pooling_subgroup_buf;
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* pooling_buf;
#endif
extern const char* winogradTransformSource2_5_1;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* unary_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* depthwise_conv2d_buf;
#endif
extern const char* glmem_convert;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* winogradTransform_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* winogradTransform_subgroup_buf;
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* splitgelu_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* select_buf;
#endif
extern const char* grid_sample;
extern const char* buffer_convert_quant;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* gemm_buf;
#endif
extern const char* conv_2d_int;
extern const char* copy_buffer_to_image2d;
extern const char* loop;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* argmax_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* buffer_convert_subgroup_buf;
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* attention_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* groupnorm_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* unary_subgroup_buf;
#endif
#endif
extern const char* gemm;
extern const char* depthwise_deconv2d;
extern const char* range;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* scale_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* matmul_buf;
#endif
extern const char* pooling;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* conv_2d_buf;
#endif
extern const char* gemm_int;
extern const char* buffer_to_image;
extern const char* winogradTransformDest2_3_1;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* layernorm_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* softmax_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* conv_2d_c16_subgroup_buf;
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* input_transe_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* reduction_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* strassen_binary_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* matmul_params_buf;
#endif
extern const char* cast;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* buffer_convert_buf;
#endif
extern const char* matmul;
extern const char* binary;
extern const char* roi_pooling;
extern const char* depthwise_conv2d;
extern const char* layernorm;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* gemm_conv1x1_buf;
#endif
extern const char* winogradTransformDest2_5_1;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* cast_buf;
#endif
extern const char* reduction;
const std::map<std::string, const char*> OpenCLProgramMap = 
 { 
  { "conv_2d", conv_2d },
  { "deconv_2d", deconv_2d },
  { "unary", unary },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "grid_sample_buf", grid_sample_buf },
#endif
  { "interp", interp },
  { "select", select },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "range_buf", range_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "self_attention_buf", self_attention_buf },
#endif
  { "winogradTransformSource2_3_1", winogradTransformSource2_3_1 },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "gemv_conv1x1_buf", gemv_conv1x1_buf },
#endif
  { "raster", raster },
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "conv_2d_c1_subgroup_buf", conv_2d_c1_subgroup_buf },
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "linear_attention_buf", linear_attention_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "matmul_local_buf", matmul_local_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "conv_2d_int_buf", conv_2d_int_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "interp_buf", interp_buf },
#endif
  { "scale", scale },
  { "softmax", softmax },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "binary_buf", binary_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "raster_buf", raster_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "binary_subgroup_buf", binary_subgroup_buf },
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "depthwise_conv2d_subgroup_buf", depthwise_conv2d_subgroup_buf },
#endif
#endif
  { "nearest", nearest },
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "pooling_subgroup_buf", pooling_subgroup_buf },
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "pooling_buf", pooling_buf },
#endif
  { "winogradTransformSource2_5_1", winogradTransformSource2_5_1 },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "unary_buf", unary_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "depthwise_conv2d_buf", depthwise_conv2d_buf },
#endif
  { "glmem_convert", glmem_convert },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "winogradTransform_buf", winogradTransform_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "winogradTransform_subgroup_buf", winogradTransform_subgroup_buf },
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "splitgelu_buf", splitgelu_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "select_buf", select_buf },
#endif
  { "grid_sample", grid_sample },
  { "buffer_convert_quant", buffer_convert_quant },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "gemm_buf", gemm_buf },
#endif
  { "conv_2d_int", conv_2d_int },
  { "copy_buffer_to_image2d", copy_buffer_to_image2d },
  { "loop", loop },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "argmax_buf", argmax_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "buffer_convert_subgroup_buf", buffer_convert_subgroup_buf },
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "attention_buf", attention_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "groupnorm_buf", groupnorm_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "unary_subgroup_buf", unary_subgroup_buf },
#endif
#endif
  { "gemm", gemm },
  { "depthwise_deconv2d", depthwise_deconv2d },
  { "range", range },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "scale_buf", scale_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "matmul_buf", matmul_buf },
#endif
  { "pooling", pooling },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "conv_2d_buf", conv_2d_buf },
#endif
  { "gemm_int", gemm_int },
  { "buffer_to_image", buffer_to_image },
  { "winogradTransformDest2_3_1", winogradTransformDest2_3_1 },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "layernorm_buf", layernorm_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "softmax_buf", softmax_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "conv_2d_c16_subgroup_buf", conv_2d_c16_subgroup_buf },
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "input_transe_buf", input_transe_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "reduction_buf", reduction_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "strassen_binary_buf", strassen_binary_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "matmul_params_buf", matmul_params_buf },
#endif
  { "cast", cast },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "buffer_convert_buf", buffer_convert_buf },
#endif
  { "matmul", matmul },
  { "binary", binary },
  { "roi_pooling", roi_pooling },
  { "depthwise_conv2d", depthwise_conv2d },
  { "layernorm", layernorm },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "gemm_conv1x1_buf", gemm_conv1x1_buf },
#endif
  { "winogradTransformDest2_5_1", winogradTransformDest2_5_1 },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "cast_buf", cast_buf },
#endif
  { "reduction", reduction },
};
}
const std::map<std::string, std::string> OpenCLProgramMd5Map = 
 { 
  { "conv_2d", "0048eb0c6a571925e6f3e1bd1d314d09" },
  { "deconv_2d", "287f563ddb48cfa1f282f50ce1bb34c2" },
  { "unary", "e122f5aca36a2e46e4b6d180c0754a34" },
  { "grid_sample_buf", "8b4bd3d6f9154b6dc6fafac3fce1c11e" },
  { "interp", "e549db5550b01a0bf984151d306cab9a" },
  { "select", "34f95349775e0610d5e6d16ae0047797" },
  { "range_buf", "427ab3d6f9ad0cafd08a0b08264df779" },
  { "self_attention_buf", "653a99a0693d9173084de43e577f9cfd" },
  { "winogradTransformSource2_3_1", "5cae94a94c822b4b6f82e238ab518f4a" },
  { "gemv_conv1x1_buf", "c345830afdfa6154633950e4a7f9e60e" },
  { "raster", "0cf8ee1f7927d0027ce0f07ad564266e" },
  { "conv_2d_c1_subgroup_buf", "04a28a410c79fa6917827d16e189f322" },
  { "linear_attention_buf", "c6995408c41a78eaaa748c4fd6e03e1a" },
  { "matmul_local_buf", "2497e20b734f5b77d021524648437b75" },
  { "conv_2d_int_buf", "6903dc7ca47d116549ac2b7c4bbf4587" },
  { "interp_buf", "2e5ff1b5184be705580ab6a221864a0c" },
  { "scale", "95773334e603db663c594945a064b9cc" },
  { "softmax", "aaa633bb6dd2c40f4379d09d754b5adc" },
  { "binary_buf", "6c5ee786c70aa485d9b49368517296e3" },
  { "raster_buf", "7e9c4013c436cc929c8e1b4d69cf1bd4" },
  { "binary_subgroup_buf", "8444f988543cd4a4d9b124442f02f999" },
  { "depthwise_conv2d_subgroup_buf", "3e37457e72b7e629655aa04bd03e559e" },
  { "nearest", "e8b2081c5e50ae6d370989f816cda543" },
  { "pooling_subgroup_buf", "9c935c0caabe2ee20822fcfd7722472e" },
  { "pooling_buf", "806c95095431e361be2af7f4e9eae65e" },
  { "winogradTransformSource2_5_1", "f0ee12556faf4fe0222e2a4e64c53c5c" },
  { "unary_buf", "bfa4bab8f0cf486ae77eb27193f03b9a" },
  { "depthwise_conv2d_buf", "d1c201a09afccebe794d50027acabdc1" },
  { "glmem_convert", "ee4866b2d889824e48d58fa3a78795d4" },
  { "winogradTransform_buf", "efa5fda527fce5820ba48b90a4707fa7" },
  { "winogradTransform_subgroup_buf", "904f2a0f1a062378418c6c90133ed5e0" },
  { "splitgelu_buf", "86d5b31ea14330d2b99273e4e868bd35" },
  { "select_buf", "1516b3f3c52ba8e8a0a5cd7f03ea86f2" },
  { "grid_sample", "0e08897ea35a57c04b834b2a83be8383" },
  { "buffer_convert_quant", "ce4ac18b78e746f7ed338f35e5237dbd" },
  { "gemm_buf", "b030b6eacaf65a54e8eabee2755f892a" },
  { "conv_2d_int", "985925b9f24d85fa38df2df9b01fafc5" },
  { "copy_buffer_to_image2d", "a72ed287711f9bb78a2cfa9726a1fa92" },
  { "loop", "4849a55cd99f0ebab72a10527455341f" },
  { "argmax_buf", "ae4a1ae3461b2758609022ac7569b11b" },
  { "buffer_convert_subgroup_buf", "d968b717e537464a7fa08e742c9a0319" },
  { "attention_buf", "7d05b22865927ca19dae5762ba6f1df9" },
  { "groupnorm_buf", "7f4b041b77ba98165ab624d94444f327" },
  { "unary_subgroup_buf", "31e3768f899da6da45084f617b13c282" },
  { "gemm", "5729018147348682e02762ed5ec14d0c" },
  { "depthwise_deconv2d", "810f69205dede9b38e4858aad621fa71" },
  { "range", "97feaf25d837a325382c162ad77ae0ca" },
  { "scale_buf", "9176b8e86fd4d326e7fa14640ce13b48" },
  { "matmul_buf", "b66faece7f0591d49c289e5227d9f680" },
  { "pooling", "900d1388836badea36a7e06ad7763b0d" },
  { "conv_2d_buf", "2faa0378ab0d702419a92ecc2073851a" },
  { "gemm_int", "41770c2a12943f8fbdbfe259889ddf2d" },
  { "buffer_to_image", "bad95040692206db84b5a1bcc0b6f248" },
  { "winogradTransformDest2_3_1", "f2aaa52d652565e70a44868d4f6028e9" },
  { "layernorm_buf", "5f6b88b29da72f51bdc85064b5663bb2" },
  { "softmax_buf", "12052d403f3fa0cdfea2559296e88e6c" },
  { "conv_2d_c16_subgroup_buf", "81f9027f323b6890d08d49dab10a15e4" },
  { "input_transe_buf", "c80482cd531add8582edc242bcbfa947" },
  { "reduction_buf", "c16506adcebf7760a1a3c96ce0d386ee" },
  { "strassen_binary_buf", "1ec57b4f87beb05457f6ef00de593d9d" },
  { "matmul_params_buf", "34fba2156345dcdb8fb07a4081a92fd1" },
  { "cast", "129055345fd1d576eb398635c81701ab" },
  { "buffer_convert_buf", "e633544642a1a9a61755c913cfe77017" },
  { "matmul", "a3e51ece4be2eb0f28266718b313c24e" },
  { "binary", "5683a6a6fd24660f0d05a70938fa6a62" },
  { "roi_pooling", "ba4a81b7ec7058d14afb377c18674a76" },
  { "depthwise_conv2d", "a23dd590e0bdcdd60987e8bab5ed529f" },
  { "layernorm", "bd457b4bd4f3c57818bc17e073b09e74" },
  { "gemm_conv1x1_buf", "5f8fd2f6e8278a2e003825a001b733ac" },
  { "winogradTransformDest2_5_1", "4f3d0d6b3e0ee7f0bff97acfbbdf653f" },
  { "cast_buf", "f39e5c1ca2fa4b39eac2af1c7934ba85" },
  { "reduction", "222cc3d09d2d5f2e0db4679a17aa8628" },
};
