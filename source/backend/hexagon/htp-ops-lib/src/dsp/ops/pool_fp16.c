#include <stdbool.h>
#include <stdint.h>

#include "dsp/hvx_utils.h"
#include "dsp/ops.h"



int hvx_pool2d_fp16(__fp16* restrict dst, const __fp16* restrict src, int batch, int ih, int iw, int oh, int ow,
                    int c4, int kernelY, int kernelX, int strideY, int strideX, int padY, int padX, int padType,
                    int countType, int poolType) {
    (void)padType;
    const int pack = __HVX_LENGTH__ / (int)sizeof(__fp16);
    if (batch <= 0 || ih <= 0 || iw <= 0 || oh <= 0 || ow <= 0 || c4 <= 0) {
        return -1;
    }

    const HVX_Vector vZero = Q6_V_vzero();

    for (int b = 0; b < batch; ++b) {
        for (int cb = 0; cb < c4; ++cb) {
            const __fp16* srcBase = src + ((size_t)(cb * batch + b) * ih * iw * pack);
            __fp16* dstBase = dst + ((size_t)(cb * batch + b) * oh * ow * pack);

            for (int oy = 0; oy < oh; ++oy) {
                int inYOrigin = oy * strideY - padY;
                for (int ox = 0; ox < ow; ++ox) {
                    int inXOrigin = ox * strideX - padX;

                    bool inited = false;
                    HVX_Vector acc = vZero;
                    int validCount = 0;

                    for (int ky = 0; ky < kernelY; ++ky) {
                        int inY = inYOrigin + ky;
                        if (inY < 0 || inY >= ih) {
                            continue;
                        }
                        for (int kx = 0; kx < kernelX; ++kx) {
                            int inX = inXOrigin + kx;
                            if (inX < 0 || inX >= iw) {
                                continue;
                            }
                            const __fp16* inPtr = srcBase + (inY * iw + inX) * pack;
                            HVX_Vector v = vmemu((const HVX_Vector*)inPtr);
                            if (poolType == 0) {
                                if (!inited) {
                                    acc = v;
                                    inited = true;
                                } else {
                                    acc = Q6_Vhf_vfmax_VhfVhf(acc, v);
                                }
                            } else {
                                if (!inited) {
                                    acc = v;
                                    inited = true;
                                } else {
                                    acc = Q6_Vhf_vadd_VhfVhf(acc, v);
                                }
                            }
                            ++validCount;
                        }
                    }

                    __fp16* outPtr = dstBase + (oy * ow + ox) * pack;
                    if (!inited) {
                        vmemu((HVX_Vector*)outPtr) = vZero;
                        continue;
                    }

                    if (poolType == 0) {
                        vmemu((HVX_Vector*)outPtr) = acc;
                    } else {
                        int denom = 1;
                        if (countType == 1) {
                            denom = kernelY * kernelX;
                        } else {
                            denom = validCount;
                        }
                        if (denom <= 0) {
                            denom = 1;
                        }
                        __fp16 inv = (__fp16)(1.0f / (float)denom);
                        HVX_Vector invV = Q6_Vh_vsplat_R(fp16_to_bits(&inv));
                        HVX_Vector out = Q6_Vhf_vmpy_VhfVhf(acc, invV);
                        vmemu((HVX_Vector*)outPtr) = out;
                    }
                }
            }
        }
    }

    return 0;
}
