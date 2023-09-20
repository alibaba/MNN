#include "ConvDepthWiseExecution.hpp"
#include "core/ConvolutionCommon.hpp"
#include "Raster.cuh"
#include <float.h>
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"

namespace MNN {
namespace CUDA {

template<typename T>
__global__ void CONV_DW(const T* input,
    const half* kernel,
    const half* bias,
    T *output,
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int dw,
    const int dh,
    const int sw,
    const int sh,
    const int pw,
    const int ph,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow,
    DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/2; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);
        
        int oz = oz_2 << 1;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];
        float color1 = bias[oz+1];

        int fxSta = max(0, (UP_DIV(-ix, dw)));
        int fySta = max(0, (UP_DIV(-iy, dh)));
        int fxEnd = min(kw, UP_DIV(iw - ix, dw));
        int fyEnd = min(kh, UP_DIV(ih - iy, dh));
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy*dh + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx*dw + ix;
                int src_offset = ((ob * ih + sy) * iw + sx) * c_p + oz;
                float inp0 = input[src_offset];
                float inp1 = input[src_offset+1];

                float ker0 = kernel[(fy * kw + fx) * c_p + oz];
                float ker1 = kernel[(fy * kw + fx) * c_p + oz + 1];

                color0 = color0 + inp0 * ker0;
                color1 = color1 + inp1 * ker1;
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        color1 = max(color1, minV);
        color1 = min(color1, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
        output[dst_offset+1] = color1;
    }
}

__global__ void CONV_DW_HALF2_OPT(const half2* input, 
    const half2* kernel, 
    const half2* bias, 
    half2 *output, 
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int dw,
    const int dh,
    const int sw,
    const int sh,
    const int pw,
    const int ph,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow,
    DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/2; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);
        
        int oz = oz_2;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        half2 color = bias[oz];

        int fxSta = max(0, -ix);
        int fySta = max(0, -iy);
        int fxEnd = min(kw, iw - ix);
        int fyEnd = min(kh, ih - iy);
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx + ix;
                int src_offset = ((ob * ih + sy) * iw + sx) * c_p + oz;
                half2 inp = input[src_offset];
                half2 ker = kernel[(fy * kw + fx) * c_p + oz];

                color = __hfma2(inp, ker, color);
            }
        }
        color.x = max(color.x, minV);
        color.x = min(color.x, maxV);

        color.y = max(color.y, minV);
        color.y = min(color.y, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;
        output[dst_offset] = color;
    }
}

__global__ void CONV_DW3x3_HALF2_OPT(const half2* input, 
    const half2* kernel, 
    const half2* bias, 
    half2 *output, 
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int dw,
    const int dh,
    const int sw,
    const int sh,
    const int pw,
    const int ph,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow,
    DivModFast d_oh
) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total/4; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox_2, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox_2);
        d_oh.divmod(tmp2, ob, oy);
        
        int oz = oz_2;
        int ox = ox_2 << 1;
        int ix = ox - 1;
        int iy = oy - 1;
        half2 color0 = bias[oz];
        half2 color1 = color0;

        half2 zero;
        zero.x = (half)0.0;
        zero.y = (half)0.0;

        half2 inp[12];
        half2 ker[3][3];
        for(int j=0; j<3; j++) {
            if(iy < 0 && j==0) {
                for(int i=0; i<4; i++) {
                    inp[i] = zero;
                }
                continue;
            }
            if(iy+2 > ih-1 && j==2) {
                for(int i=0; i<4; i++) {
                    inp[8+i] = zero;
                }
                continue;
            }

            for(int i=0; i<4; i++) {
                if(ix < 0 && i==0) {
                    for(int j=0; j<3; j++) {
                        inp[4*j+0] = zero;
                    }
                    continue;
                }
                if(ix+3 > iw-1 && i==3) {
                    for(int j=0; j<3; j++) {
                        inp[4*j+3] = zero;
                    }
                    continue;
                }
                int src_offset = ((ob * ih + iy+j) * iw + ix+i) * c_p + oz;
                inp[4*j+i] = input[src_offset];
            }
        }

        for(int j=0; j<3; j++) {
            for(int i=0; i<3; i++) {
                ker[j][i] = kernel[(j * 3 + i) * c_p + oz];
            }
        }

        for(int j=0; j<3; j++) {
            for(int i=0; i<3; i++) {
                color0 = __hfma2(inp[4*j+i], ker[j][i], color0);
                color1 = __hfma2(inp[4*j+i+1], ker[j][i], color1);
            }
        }

        color0.x = max(color0.x, minV);
        color0.x = min(color0.x, maxV);
        color0.y = max(color0.y, minV);
        color0.y = min(color0.y, maxV);

        color1.x = max(color1.x, minV);
        color1.x = min(color1.x, maxV);
        color1.y = max(color1.y, minV);
        color1.y = min(color1.y, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;
        output[dst_offset] = color0;
        output[dst_offset+c_p] = color1;
    }
}

__global__ void CONV_DW_OPT(const float* input, const half* kernel, const half* bias, float *output,
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int dw,
    const int dh,
    const int sw,
    const int sh,
    const int pw,
    const int ph,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow,
    DivModFast d_oh
    ) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total / 2; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox, tmp1, ob;
        d_oc.divmod(index, tmp1, oz_2);
        d_ow.divmod(tmp1, tmp2, ox);
        d_oh.divmod(tmp2, ob, oy);

        int oz = oz_2 << 1;
        int ix = ox * sw - pw;
        int iy = oy * sh - ph;
        float color0 = bias[oz];
        float color1 = bias[oz+1];

        int fxSta = max(0, -ix);
        int fySta = max(0, -iy);
        int fxEnd = min(kw, iw - ix);
        int fyEnd = min(kh, ih - iy);
        int fx, fy, fz;
        for (fy=fySta; fy<fyEnd; ++fy) {
            int sy = fy + iy;
            for (fx=fxSta; fx<fxEnd; ++fx) {
                int sx = fx + ix;
                int src_offset = ((ob * ih + sy) * iw + sx) * c_p + oz;
                float inp0 = input[src_offset];
                float inp1 = input[src_offset+1];

                float ker0 = kernel[(fy * kw + fx) * c_p + oz];
                float ker1 = kernel[(fy * kw + fx) * c_p + oz + 1];

                color0 = color0 + inp0 * ker0;
                color1 = color1 + inp1 * ker1;
            }
        }
        color0 = max(color0, minV);
        color0 = min(color0, maxV);

        color1 = max(color1, minV);
        color1 = min(color1, maxV);

        int dst_offset = ((ob * oh + oy) * ow + ox) * c_p + oz;

        output[dst_offset] = color0;
        output[dst_offset+1] = color1;
    }
}

template<typename T>
__global__ void CONV_DW_MULTI_WIDTH4(const T* input, const half* kernel, const half* bias, T *output,
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int total,
    DivModFast d_oc,
    DivModFast d_ow_4,
    DivModFast d_oh
    ) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total / 4; index += blockDim.x * gridDim.x) {
        int oz, tmp2, oy, ox_4, tmp1, ob;
        d_oc.divmod(index, tmp1, oz);
        d_ow_4.divmod(tmp1, tmp2, ox_4);
        d_oh.divmod(tmp2, ob, oy);

        float color0 = bias[oz];
        float color1 = color0;
        float color2 = color0;
        float color3 = color0;

        // Parallel pipelining read and calculate
        float src; 
        float filter0, filter1, filter2, filter3;
        int src_offset = ((ob * ih + oy) * iw + (ox_4 << 2)) * c_p + oz;
        int filter_offset = 0 * c_p + oz;

        src    = input[src_offset + 0 * c_p];
        filter0 = kernel[filter_offset + 0 * c_p];
        color0 += (src * filter0);

        filter1 = kernel[filter_offset + 1 * c_p];
        src    = input[src_offset + 1 * c_p];
        color0 += (src * filter1);
        color1 += (src * filter0);

        filter2 = kernel[filter_offset + 2 * c_p];
        src    = input[src_offset + 2 * c_p];
        color0 += (src * filter2);
        color1 += (src * filter1);
        color2 += (src * filter0);

        filter3 = kernel[filter_offset + 3 * c_p];



        for (int fx=3; fx<kw; ++fx) {
            src    = input[src_offset + fx * c_p];
            color0 += (src * filter3);
            color1 += (src * filter2);
            color2 += (src * filter1);
            color3 += (src * filter0);

            filter0 = filter1;
            filter1 = filter2;
            filter2 = filter3;
            filter3 = kernel[filter_offset + (fx+1) * c_p];
        }

        src    = input[src_offset + kw * c_p];
        color1 += (src * filter2);
        color2 += (src * filter1);
        color3 += (src * filter0);

        src    = input[src_offset + (kw+1) * c_p];
        color2 += (src * filter2);
        color3 += (src * filter1);

        src    = input[src_offset + (kw+2) * c_p];
        color3 += (src * filter2);


        color0 = max(color0, minV);
        color0 = min(color0, maxV);
        color1 = max(color1, minV);
        color1 = min(color1, maxV);

        color2 = max(color2, minV);
        color2 = min(color2, maxV);
        color3 = max(color3, minV);
        color3 = min(color3, maxV);

        int dst_offset = ((ob * oh + oy) * ow + (ox_4 << 2)) * c_p + oz;

        output[dst_offset] = color0;
        output[dst_offset+c_p] = color1;
        output[dst_offset+2*c_p] = color2;
        output[dst_offset+3*c_p] = color3;
    }
}

__global__ void CONV_DW_MULTI_WIDTH_CHANNEL(const float* input, const half* kernel, const half* bias, float *output,
    const float maxV,
    const float minV,
    const int iw,
    const int ih,
    const int c,
    const int c_p,
    const int ow,
    const int oh,
    const int kw,
    const int kh,
    const int total,
    DivModFast d_oc_2,
    DivModFast d_ow_2,
    DivModFast d_oh
    ) {

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total / 4; index += blockDim.x * gridDim.x) {
        int oz_2, tmp2, oy, ox_2, tmp1, ob;
        d_oc_2.divmod(index, tmp1, oz_2);
        d_ow_2.divmod(tmp1, tmp2, ox_2);
        d_oh.divmod(tmp2, ob, oy);

        float2 color0 =  __half22float2((( half2 *)(bias + (oz_2 << 1)))[0]);
        float2 color1 = color0;

        // Parallel pipelining read and calculate
        float src0, src2, filter0, filter2;
        int src_offset = ((ob * ih + oy) * iw + (ox_2 << 1)) * c_p + (oz_2 << 1);
        int filter_offset = 0 * c_p + (oz_2 << 1);

        float2 src    = ((float2 *)(input + src_offset + 0 * c_p))[0];
        float2 filter = __half22float2(((half2 *)(kernel + filter_offset + 0 * c_p))[0]);
        
        color0.x += (src.x * filter.x);
        color0.y += (src.y * filter.y);

        for (int fx=1; fx<kw; ++fx) {
            src    = ((float2 *)(input + src_offset + fx * c_p))[0];
            color1.x += (src.x * filter.x);
            color1.y += (src.y * filter.y);

            filter = __half22float2(((half2 *)(void *)(kernel + filter_offset + fx * c_p))[0]);
            color0.x += (src.x * filter.x);
            color0.y += (src.y * filter.y);
        }

        src    = ((float2 *)(input + src_offset + kw * c_p))[0];
        color1.x += (src.x * filter.x);
        color1.y += (src.y * filter.y);

        color0.x = max(color0.x, minV);
        color0.x = min(color0.x, maxV);
        color1.x = max(color1.x, minV);
        color1.x = min(color1.x, maxV);

        color0.y = max(color0.y, minV);
        color0.y = min(color0.y, maxV);
        color1.y = max(color1.y, minV);
        color1.y = min(color1.y, maxV);

        int dst_offset = ((ob * oh + oy) * ow + (ox_2 << 1)) * c_p + (oz_2 << 1);

        ((float2 *)(output + dst_offset))[0] = color0;
        ((float2 *)(output + dst_offset + c_p))[0] = color1;
    }
}

ErrorCode ConvDepthWiseCompute(Backend* bn,
                               const int blockNum,
                               const int threadNum,
                               const void * inputAddr,
                               const void * filterAddr,
                               const void * biasAddr,
                               void * outputAddr,
                               const float maxV,
                               const float minV,
                               const int iw,
                               const int ih,
                               const int c,
                               const int c_p,
                               const int ow,
                               const int oh,
                               const int kw,
                               const int kh,
                               const int dw,
                               const int dh,
                               const int sw,
                               const int sh,
                               const int pw,
                               const int ph,
                               const int total,
                               DivModFast d_oc,
                               DivModFast d_ow,
                               DivModFast d_oh) {

    #ifdef ENABLE_CUDA_BF16
    if (static_cast<CUDABackend*>(bn)->getPrecision() == 3) {
        if(kw==3 && kh==3 && sw==1 && sh==1 && pw==1 && ph==1 && ow % 2 ==0) {
            DivModFast d_ow2(ow/2);
            CONV_DW3x3_BF162_OPT<<<blockNum, threadNum>>>((const __nv_bfloat162*)inputAddr, (const __nv_bfloat162*)filterAddr,
                (const __nv_bfloat162*)biasAddr, (__nv_bfloat162*)outputAddr,
                maxV, minV, iw, ih, c, c_p / 2, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
                d_oc, d_ow2, d_oh);
            checkKernelErrors;
            return NO_ERROR;
        }
        if(dw == 1 && dh == 1) {
            if(sw == 1 && sh == 1 && pw == 0 && ph == 0 && kw > 3 && kw < 12 && kh == 1 && pw == 0 && ph == 0 && ow % 4 == 0) {
                DivModFast d_oc(c * PACK_NUMBER);
                DivModFast d_ow(ow/4);
                CONV_DW_BF16_MULTI_WIDTH4<<<blockNum, threadNum>>>((const __nv_bfloat16*)inputAddr, (const __nv_bfloat16*)filterAddr,
                    (const __nv_bfloat16*)biasAddr, (__nv_bfloat16*)outputAddr,
                    maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, total,
                    d_oc, d_ow, d_oh);
                checkKernelErrors;
            } else {
                CONV_DW_BF162_OPT<<<blockNum, threadNum>>>((const __nv_bfloat162*)inputAddr, (const __nv_bfloat162*)filterAddr,
                    (const __nv_bfloat162*)biasAddr, (__nv_bfloat162*)outputAddr,
                    maxV, minV, iw, ih, c, c_p / 2, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
                    d_oc, d_ow, d_oh);
                checkKernelErrors;
            }
        } else {
            CONV_DW_BF16<<<blockNum, threadNum>>>((const __nv_bfloat16*)inputAddr, (const __nv_bfloat16*)filterAddr,
                (const __nv_bfloat16*)biasAddr, (__nv_bfloat16*)outputAddr,
                maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
                d_oc, d_ow, d_oh);
            checkKernelErrors;
        }
        return NO_ERROR;

    }
    #endif

    if (static_cast<CUDABackend*>(bn)->useFp16()) {
        if(kw==3 && kh==3 && sw==1 && sh==1 && pw==1 && ph==1 && ow % 2 ==0) {
            DivModFast d_ow2(ow/2);

            CONV_DW3x3_HALF2_OPT<<<blockNum, threadNum>>>((const half2*)inputAddr, (const half2*)filterAddr,
                (const half2*)biasAddr, (half2*)outputAddr,
                maxV, minV, iw, ih, c, c_p / 2, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
                d_oc, d_ow2, d_oh);
            checkKernelErrors;
            return NO_ERROR;
        }
        if(dw == 1 && dh == 1) {
            if(sw == 1 && sh == 1 && pw == 0 && ph == 0 && kw > 3 && kw < 12 && kh == 1 && pw == 0 && ph == 0 && ow % 4 == 0) {
                DivModFast d_oc(c * PACK_NUMBER);
                DivModFast d_ow(ow/4);
                CONV_DW_MULTI_WIDTH4<<<blockNum, threadNum>>>((const half*)inputAddr, (const half*)filterAddr,
                    (const half*)biasAddr, (half*)outputAddr,
                    maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, total,
                    d_oc, d_ow, d_oh);
                checkKernelErrors;
            } else {
                CONV_DW_HALF2_OPT<<<blockNum, threadNum>>>((const half2*)inputAddr, (const half2*)filterAddr,
                    (const half2*)biasAddr, (half2*)outputAddr,
                    maxV, minV, iw, ih, c, c_p / 2, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
                    d_oc, d_ow, d_oh);//_HALF_OPT
                checkKernelErrors;
            }
        } else {
            CONV_DW<<<blockNum, threadNum>>>((const half*)inputAddr, (const half*)filterAddr,
                (const half*)biasAddr, (half*)outputAddr,
                maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
                d_oc, d_ow, d_oh);
            checkKernelErrors;
        }
        return NO_ERROR;
    }

    if(dw == 1 && dh == 1) { 
        if(sw == 1 && sh == 1 && pw == 0 && ph == 0 && kw > 3 && kw < 12 && kh == 1 && pw == 0 && ph == 0) {
            
            if(ow % 4 == 0) {
                DivModFast d_oc(c * PACK_NUMBER);
                DivModFast d_ow(ow/4);
                CONV_DW_MULTI_WIDTH4<<<blockNum, threadNum>>>((const float*)inputAddr, (const half*)filterAddr,
                    (const half*)biasAddr, (float*)outputAddr,
                    maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, total,
                    d_oc, d_ow, d_oh);
                checkKernelErrors;
            } else if(ow % 2 == 0) {
                DivModFast d_oc(c * PACK_NUMBER / 2);
                DivModFast d_ow(ow/2);
                CONV_DW_MULTI_WIDTH_CHANNEL<<<blockNum, threadNum>>>((const float*)inputAddr, (const half*)filterAddr,
                    (const half*)biasAddr, (float*)outputAddr,
                    maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, total,
                    d_oc, d_ow, d_oh);
                checkKernelErrors;
            } else {
                CONV_DW_OPT<<<blockNum, threadNum>>>((const float*)inputAddr, (const half*)filterAddr,
                    (const half*)biasAddr, (float*)outputAddr,
                    maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
                    d_oc, d_ow, d_oh);
                checkKernelErrors;
    }
        } else {
            CONV_DW_OPT<<<blockNum, threadNum>>>((const float*)inputAddr, (const half*)filterAddr,
                (const half*)biasAddr, (float*)outputAddr,
                maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
                d_oc, d_ow, d_oh);
            checkKernelErrors;
        }
    } else {
        CONV_DW<<<blockNum, threadNum>>>((const float*)inputAddr, (const half*)filterAddr,
            (const half*)biasAddr, (float*)outputAddr,
            maxV, minV, iw, ih, c, c_p, ow, oh, kw, kh, dw, dh, sw, sh, pw, ph, total,
            d_oc, d_ow, d_oh);
        checkKernelErrors;
    }

    return NO_ERROR;

}

static std::shared_ptr<ConvDepthWiseExecution::Resource> _makeResource(const Op* op, Backend* bn) {
    std::shared_ptr<ConvDepthWiseExecution::Resource> res(new ConvDepthWiseExecution::Resource);
    auto pool = static_cast<CUDABackend*>(bn)->getStaticBufferPool();
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();
    auto conv = op->main_as_Convolution2D();
    auto convCommon = conv->common();
    int kernelX = convCommon->kernelX();
    int kernelY = convCommon->kernelY();
    int depth = convCommon->outputCount();
    int depthC = UP_DIV(depth, PACK_NUMBER);
    res->weightTensor.reset(Tensor::createDevice<float>({kernelX * kernelY * depthC * PACK_NUMBER}));
    bool success = bn->onAcquireBuffer(res->weightTensor.get(), Backend::STATIC);
    if (!success) {
        return nullptr;
    }
    res->mFilter = (void *)res->weightTensor.get()->buffer().device;

    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, bn, conv, &filterDataPtr, &weightSize);
    auto tempWeightStorage = pool->alloc(depthC * PACK_NUMBER * kernelY * kernelX * sizeof(float));
    auto tempWeight = (uint8_t*)tempWeightStorage.first + tempWeightStorage.second;
    cuda_check(cudaMemset(tempWeight, 0, depthC * PACK_NUMBER * kernelY * kernelX * sizeof(float)));
    cuda_check(cudaMemcpy(tempWeight, filterDataPtr, weightSize*sizeof(float), cudaMemcpyHostToDevice));

    FuseRegion reg;
    int offset[8 * PACK_NUMBER];
    auto regionStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(sizeof(FuseRegion));
    auto offsetGpuStorage = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(sizeof(offset));
    auto offsetGpu = (uint8_t*)offsetGpuStorage.first + offsetGpuStorage.second;
    
    #ifdef ENABLE_CUDA_BF16
    if(static_cast<CUDABackend*>(bn)->getPrecision() == 3) {
        // [Oc, Kh*Kw] -> [Kh*Kw, Oc(p)]
        DivModFast d_ocp(depthC * PACK_NUMBER);
        auto count =  depthC * PACK_NUMBER * kernelY * kernelX;
        int block_num = runtime->blocks_num(count);
        int threads_num = runtime->threads_num();
        WeightTransToBf16<<<block_num, threads_num>>>((const float*)tempWeight, (__nv_bfloat16*)res->mFilter, count,\
            kernelY * kernelX, depth, d_ocp);
        checkKernelErrors;
    } 
    else
    #endif
    {
        reg.size[0] = 1;
        reg.size[1] = kernelY * kernelX;
        reg.size[2] = depthC * PACK_NUMBER;
        reg.srcStride[0] = 0;
        reg.srcStride[1] = 1;
        reg.srcStride[2] = kernelY * kernelX;
        reg.dstStride[0] = 0;
        reg.dstStride[1] = depthC * PACK_NUMBER;
        reg.dstStride[2] = 1;
        offset[0] = 1;
        offset[1] = kernelY * kernelX;
        offset[2] = depth;
        offset[3] = 0;
        offset[4] = 1;
        offset[5] = reg.size[1];
        offset[6] = reg.size[2];
        offset[7] = 0;
        reg.fuseNumber = 1;

        runtime->memcpy((uint8_t*)regionStorage.first + regionStorage.second, &reg, sizeof(FuseRegion), MNNMemcpyHostToDevice, true);
        runtime->memcpy(offsetGpu, offset, 8 * sizeof(int), MNNMemcpyHostToDevice, true);
        FuseRasterBlitFloatToHalf((uint8_t*)res->mFilter, (uint8_t*)tempWeight, (FuseRegion*)((uint8_t*)regionStorage.first + regionStorage.second), offsetGpu, runtime);
    }
    pool->free(tempWeightStorage);
    res->biasTensor.reset(Tensor::createDevice<float>({depthC * PACK_NUMBER}));
    success = bn->onAcquireBuffer(res->biasTensor.get(), Backend::STATIC);
    res->mBias = (void *)res->biasTensor.get()->buffer().device;
    if (!success) {
        return nullptr;
    }
    if(conv->bias() != nullptr) {
        auto tempBiasStorage = pool->alloc(depth * sizeof(float));
        auto tempBias = (uint8_t*)tempBiasStorage.first + tempBiasStorage.second;
        cuda_check(cudaMemcpy(tempBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

        #ifdef ENABLE_CUDA_BF16
        if(static_cast<CUDABackend*>(bn)->getPrecision() == 3) 
        {
            auto countBias = depthC * PACK_NUMBER;
            int block_num = runtime->blocks_num(countBias);
            int threads_num = runtime->threads_num();
            BiasTransToBf16<<<block_num, threads_num>>>((const float*)tempBias, (__nv_bfloat16*)res->mBias, countBias, depth);
            checkKernelErrors;
        } 
        else 
        #endif
        {
            reg.size[0] = 1;
            reg.size[1] = 1;
            reg.size[2] = depthC * PACK_NUMBER;
            reg.srcStride[0] = 0;
            reg.srcStride[1] = 0;
            reg.srcStride[2] = 1;
            reg.dstStride[0] = 0;
            reg.dstStride[1] = 0;
            reg.dstStride[2] = 1;
            offset[0] = 1;
            offset[1] = 1;
            offset[2] = conv->bias()->size();
            offset[3] = 0;
            offset[4] = 1;
            offset[5] = 1;
            offset[6] = reg.size[2];
            offset[7] = 0;
            reg.fuseNumber = 1;
            runtime->memcpy((uint8_t*)regionStorage.first + regionStorage.second, &reg, sizeof(FuseRegion), MNNMemcpyHostToDevice, true);
            runtime->memcpy(offsetGpu, offset, 8 * sizeof(int), MNNMemcpyHostToDevice, true);
            FuseRasterBlitFloatToHalf((uint8_t*)res->mBias, (uint8_t*)tempBias, (FuseRegion*)((uint8_t*)regionStorage.first + regionStorage.second), offsetGpu, runtime);
        }
        pool->free(tempBiasStorage);
    }
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(regionStorage);
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(offsetGpuStorage);
    return res;
}

ConvDepthWiseExecution::ConvDepthWiseExecution(const Op* op, Backend* bn, std::shared_ptr<Resource> resource) : Execution(bn) {
    mOp = op;
    mResource = resource;
}

ConvDepthWiseExecution::~ ConvDepthWiseExecution() {
    //
}

ErrorCode ConvDepthWiseExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto pad = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], mOp->main_as_Convolution2D()->common());
    auto conv = mOp->main_as_Convolution2D();
    auto convCommon = mOp->main_as_Convolution2D()->common();
    int channel = inputs[0]->channel();
    int channelDiv = UP_DIV(channel, PACK_NUMBER);
    parameters.pad[0] = pad.first;
    parameters.pad[1] = pad.second;
    parameters.kernelSize[0] = convCommon->kernelX();
    parameters.kernelSize[1] = convCommon->kernelY();
    parameters.stride[0] = convCommon->strideX();
    parameters.stride[1] = convCommon->strideY();
    parameters.dilate[0] = convCommon->dilateX();
    parameters.dilate[1] = convCommon->dilateY();
    parameters.inputSize[0] = inputs[0]->width();
    parameters.inputSize[1] = inputs[0]->height();
    parameters.channel = channelDiv;
    parameters.outputSize[0] = outputs[0]->width();
    parameters.outputSize[1] = outputs[0]->height();
    parameters.batch = inputs[0]->batch();

    parameters.total = parameters.batch * parameters.outputSize[1] * parameters.outputSize[0] * parameters.channel * PACK_NUMBER;
    if (static_cast<CUDABackend*>(backend())->useFp16()) {
        // Do nothing
    } else {
        parameters.minValue = -FLT_MAX;
        parameters.maxValue = FLT_MAX;
    }
    if (convCommon->relu()) {
        parameters.minValue = 0.0f;
    }
    if (convCommon->relu6()) {
        parameters.minValue = 0.0f;
        parameters.maxValue = 6.0f;
    }
    mTotalCount = parameters.total;
    //MNN_PRINT("%d-%d-%d-%d, %d-%d-%d-%d-%d\n", parameters.kernelSize[0], parameters.kernelSize[1], parameters.stride[0], parameters.stride[1], parameters.inputSize[0], parameters.inputSize[1], channel, parameters.outputSize[0], parameters.outputSize[1]);
    return NO_ERROR;
}

ErrorCode ConvDepthWiseExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto& prop = runtime->prop();
    int limitThreads = UP_DIV(mTotalCount, prop.multiProcessorCount);
    int threadNum = ALIMIN(prop.maxThreadsPerBlock/2, limitThreads);
    int blockNum = prop.multiProcessorCount;

    const float maxV = parameters.maxValue;
    const float minV = parameters.minValue;
    const int iw = parameters.inputSize[0];
    const int ih = parameters.inputSize[1];
    const int c = parameters.channel;
    const int c_p = c * PACK_NUMBER;
    const int ow = parameters.outputSize[0];
    const int oh = parameters.outputSize[1];
    const int kw = parameters.kernelSize[0];
    const int kh = parameters.kernelSize[1];
    const int dw = parameters.dilate[0];
    const int dh = parameters.dilate[1];
    const int sw = parameters.stride[0];
    const int sh = parameters.stride[1];
    const int pw = parameters.pad[0];
    const int ph = parameters.pad[1];
    const int total = parameters.total;

    DivModFast d_oc(parameters.channel * PACK_NUMBER / 2);
    DivModFast d_ow(parameters.outputSize[0]);
    DivModFast d_oh(parameters.outputSize[1]);

    ErrorCode res = ConvDepthWiseCompute(backend(),
                                         blockNum,
                                         threadNum,
                                         (const void *)inputs[0]->deviceId(),
                                         mResource->mFilter,
                                         mResource->mBias,
                                         (void *)outputs[0]->deviceId(),
                                         maxV,
                                         minV,
                                         iw,
                                         ih,
                                         c,
                                         c_p,
                                         ow,
                                         oh,
                                         kw,
                                         kh,
                                         dw,
                                         dh,
                                         sw,
                                         sh,
                                         pw,
                                         ph,
                                         total,
                                         d_oc,
                                         d_ow,
                                         d_oh);

    return res;

}

class ConvDepthWiseExecutionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (inputs.size() > 1) {
            return new MultiInputConvDepthWiseExecution(op, backend);
        }
        auto res = _makeResource(op, backend);
        if (nullptr == res) {
            return nullptr;
        }
        return new ConvDepthWiseExecution(op, backend, res);
    }
};

static CUDACreatorRegister<ConvDepthWiseExecutionCreator> __init(OpType_ConvolutionDepthwise);
}
}
