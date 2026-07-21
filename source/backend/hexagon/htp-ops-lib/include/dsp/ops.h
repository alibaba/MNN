#pragma once

#include <stdint.h>

#ifndef restrict
#  define restrict __restrict
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TEST_M_PACK 32
#define TEST_N_PACK 32
#define TEST_K_PACK 32

#define __vtcm  // only a hint, no real effect

int hmx_matmulq4fp16(uint8_t * c, const uint8_t * a, const uint8_t * b, const uint8_t * b_scale, const uint8_t * bias, int m, int k, int n, int mp, int np, int kp, int scale_block_num, int scale_asymmetric);
int hmx_matmulq4fp16_mle32(uint8_t * c, const uint8_t * a, const uint8_t * b, const uint8_t * b_scale, const uint8_t * bias, int m, int k, int n, int mp, int np, int kp, int scale_block_num, int scale_asymmetric);
int hmx_matmulq4blockfp16_mle32(uint8_t * c, const uint8_t * a, const uint8_t * b, const uint8_t * b_scale, const uint8_t * bias, int m, int k, int n, int mp, int np, int kp, int scale_block_num, int scale_asymmetric);
int scalar_tmac_a16w1_fp16(uint8_t *dst, const uint8_t *src, const uint8_t *weight, const float *scale,
                           const uint8_t *bias, int m, int ic, int oc, int scale_block_num,
                           int scale_asymmetric, int relu, int relu6, int output_bytes);
typedef struct Im2ColParameter {
    int32_t padX;
    int32_t padY;
    int32_t dilateX;
    int32_t dilateY;
    int32_t strideX;
    int32_t strideY;
    int32_t kernelX;
    int32_t kernelY;
    int32_t icDiv4;
    int32_t kernelCountUnit;
    int32_t iw;
    int32_t ih;
    int32_t ow;
    int32_t oh;
    int32_t srcZStep;
    int32_t srcYStep;
    int32_t packCUnit;
    int32_t destICStride;
    int32_t ic;
    int32_t icup4;
} Im2ColParameter;

typedef struct HmxIm2ColConvParam {
    Im2ColParameter im2col;
    int32_t oc;
    int32_t mp;
    int32_t np;
    int32_t relu;
    int32_t relu6;
    int32_t batch;
    int32_t outputBytes;
} HmxIm2ColConvParam;

typedef struct WeightReorderParam {
    int32_t ic;
    int32_t oc;
    int32_t kernelX;
    int32_t kernelY;
} WeightReorderParam;

int hmx_im2col_convolution_fp16(uint8_t *dst, const uint8_t *src, const uint8_t *weight, const uint8_t *bias,
                                const HmxIm2ColConvParam* params);
int htp_ops_conv1x1_direct_fp16(uint8_t* output, uint8_t* input, uint8_t* weight, uint8_t* bias,
                                const HmxIm2ColConvParam* params);
int hvx_pool2d_fp16(__fp16 *restrict dst, const __fp16 *restrict src,
                   int batch, int ih, int iw, int oh, int ow, int c4,
                   int kernelY, int kernelX, int strideY, int strideX,
                   int padY, int padX, int padType, int countType, int poolType);

int hvx_conv_depthwise2d_fp16(__fp16 *restrict dst, const __fp16 *restrict src,
                              const __fp16 *restrict weight, const __fp16 *restrict bias,
                              int batch, int ih, int iw, int oh, int ow, int c4,
                              int kernelY, int kernelX, int strideY, int strideX,
                              int padY, int padX, int dilateY, int dilateX, int relu, int relu6);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace op_utils {

int compare_result(const float *x, const float *y, int n_elems);

}

#endif
