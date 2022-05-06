#include "ImageColumn.cuh"
#include "MNNCUDADefine.hpp"
#include "MNNCUDAFunction.cuh"
#include "Raster.cuh"

#define BLOCK_INT4 2

namespace MNN {
namespace CUDA {

__global__ void Im2Col1x1(const ConvolutionCommon::Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const float* A,
    half* AP,
    const int ePack,
    const int eShift,
    DivModFast eAlignD,
    DivModFast owD,
    DivModFast ohD
    ) {
    int eAlign = matmulParam->elhPack[0] * ePack;
    int lAlign = matmulParam->elhPack[1];
    int maxCount = eAlign * lAlign * BLOCK_INT4;
    int kernelCount = 1;
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int index = indexO >> 1;
        int lR = indexO & 1;
        int eIndex, lIndex;
        eAlignD.divmod(index, lIndex, eIndex);
        int eU = eIndex >> eShift;
        int eR = eIndex & (ePack-1);
        int dstOffset = eU * matmulParam->elhPack[1] * (ePack * MATMULPACK) + lIndex * (ePack * MATMULPACK) + eR * MATMULPACK + lR * 8;
        int4* dst = (int4*)(AP + dstOffset);
        if (eIndex >= matmulParam->elh[0]) {
            *dst = {0, 0, 0, 0};
            continue;
        }
        // Compute for source
        int ox, oy, ob;
        owD.divmod(eIndex, oy, ox);
        ohD.divmod(oy, ob, oy);
        int sz = lIndex;
        int sx = ox * param->strideX - param->padX;
        int sy = oy * param->strideY - param->padY;
        if (sx >= 0 && sx < param->iw) {
            if (sy >=0 && sy < param->ih) {
                int offset = sz * param->srcZStep + (ob * param->iw * param->ih + sy * param->iw + sx) * PACK_NUMBER + lR * 8;
                float2* srcF = (float2*)(A + offset);
                half2* dstH = (half2*)dst;
                dstH[0] = __float22half2_rn(srcF[0]);
                dstH[1] = __float22half2_rn(srcF[1]);
                dstH[2] = __float22half2_rn(srcF[2]);
                dstH[3] = __float22half2_rn(srcF[3]);
                continue;
            }
        }
        *dst = {0, 0, 0, 0};
    }
}

__global__ void Im2Col1x1_OPT(const ConvolutionCommon::Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const int maxCount, 
    const float* A,
    half* AP,
    const int ePack,
    const int eShift,
    DivModFast eAlignD,
    DivModFast owD,
    DivModFast ohD
    ) {
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int index = indexO >> 3;
        int lR = indexO & 7;
        int eIndex, lIndex;
        eAlignD.divmod(index, lIndex, eIndex);
        int eU = eIndex >> eShift;
        int eR = eIndex & (ePack-1);
        int dstOffset = ((eU * matmulParam->elhPack[1] + lIndex) << (4+eShift)) + (eR << 4) + (lR << 1);

        int offset = lIndex * param->srcZStep + (eIndex << 4) + (lR << 1);
        float2* srcF = (float2*)(A + offset);
        half2* dstH = (half2*)(AP + dstOffset);
        dstH[0] = __float22half2_rn(srcF[0]);
    }
}

__global__ void Im2Col(const ConvolutionCommon::Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const int iBlock,
    const float* A,
    half* AP,
    const int ePack,
    const int eShift
) {
    int eAlign = matmulParam->elhPack[0] * ePack;
    int lAlign = matmulParam->elhPack[1];
    int maxCount = eAlign * lAlign * BLOCK_INT4;
    int kernelCount = param->kernelX * param->kernelY;
    for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
        int index = indexO / BLOCK_INT4;
        int lR = indexO % BLOCK_INT4;
        int eIndex = index % eAlign;
        int lIndex = index / eAlign;
        int eU = eIndex / ePack;
        int eR = eIndex % ePack;
        int dstOffset = eU * matmulParam->elhPack[1] * (ePack * MATMULPACK) + lIndex * (ePack * MATMULPACK) + eR * MATMULPACK + lR * 8;
        int4* dst = (int4*)(AP + dstOffset);

        eIndex += (iBlock*matmulParam->elhPack[0]*ePack);
        if (eIndex >= matmulParam->elh[0]) {
            *dst = {0, 0, 0, 0};
            continue;
        }
        // Compute for source
        int ox = eIndex % param->ow;
        int oy = eIndex / param->ow;
        int ob = oy / param->oh;
        oy = oy % param->oh;
        int sz = lIndex / kernelCount;
        int kI = lIndex % kernelCount;
        int ksx = kI % param->kernelX;
        int ksy = kI / param->kernelX;

        int sx = ox * param->strideX + ksx * param->dilateX - param->padX;
        int sy = oy * param->strideY + ksy * param->dilateY - param->padY;
        if (sx >= 0 && sx < param->iw) {
            if (sy >=0 && sy < param->ih) {
                int offset = sz * param->srcZStep + (ob * param->iw * param->ih + sy * param->iw + sx) * PACK_NUMBER + lR * 8;
                float2* srcF = (float2*)(A + offset);
                half2* dstH = (half2*)dst;
                dstH[0] = __float22half2_rn(srcF[0]);
                dstH[1] = __float22half2_rn(srcF[1]);
                dstH[2] = __float22half2_rn(srcF[2]);
                dstH[3] = __float22half2_rn(srcF[3]);
                continue;
            }
        }
        *dst = {0, 0, 0, 0};
    }
}

__global__ void Im2Col1x1_half(const ConvolutionCommon::Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const half* A,
    half* AP,
    const int ePack,
    const int eShift,
    DivModFast eAlignD,
    DivModFast owD,
    DivModFast ohD
    ) {
int eAlign = matmulParam->elhPack[0] * ePack;
int lAlign = matmulParam->elhPack[1];
int maxCount = eAlign * lAlign * BLOCK_INT4;
int kernelCount = 1;
for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
    int index = indexO / BLOCK_INT4;
    int lR = indexO % BLOCK_INT4;
    int eIndex, lIndex;
    eAlignD.divmod(index, lIndex, eIndex);
    int eU = eIndex / ePack;
    int eR = eIndex % ePack;
    int dstOffset = eU * matmulParam->elhPack[1] * (ePack * MATMULPACK) + lIndex * (ePack * MATMULPACK) + eR * MATMULPACK + lR * 8;
    int4* dst = (int4*)(AP + dstOffset);
    if (eIndex >= matmulParam->elh[0]) {
        *dst = {0, 0, 0, 0};
        continue;
    }
    // Compute for source
    int ox, oy, ob;
    owD.divmod(eIndex, oy, ox);
    ohD.divmod(oy, ob, oy);
    int sz = lIndex;
    int sx = ox * param->strideX - param->padX;
    int sy = oy * param->strideY - param->padY;
    if (sx >= 0 && sx < param->iw) {
        if (sy >=0 && sy < param->ih) {
            int offset = sz * param->srcZStep + (ob * param->iw * param->ih + sy * param->iw + sx) * PACK_NUMBER + lR * 8;
            int4* src = (int4*)(A + offset);
            *dst = *src;
            continue;
        }
    }
    *dst = {0, 0, 0, 0};
}
}

__global__ void Im2Col1x1_half_OPT(const ConvolutionCommon::Im2ColParameter* param,
const MatMulParam* matmulParam,
const int maxCount, 
const half* A,
half* AP,
const int ePack,
const int eShift,
DivModFast eAlignD,
DivModFast owD,
DivModFast ohD
) {
for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
    size_t index = indexO >> 3;
    size_t lR = indexO & 7;
    int eIndex, lIndex;
    eAlignD.divmod(index, lIndex, eIndex);
    size_t eU = eIndex >> eShift;
    size_t eR = eIndex & (ePack-1);
    size_t dstOffset = ((eU * (size_t)matmulParam->elhPack[1] + (size_t)lIndex) << (4+eShift)) + (eR << 4) + (lR << 1);

    size_t offset = (size_t)lIndex * (size_t)param->srcZStep + ((size_t)eIndex << 4) + (lR << 1);
    int* srcF = (int*)(A + offset);
    int* dstH = (int*)(AP + dstOffset);
    dstH[0] = srcF[0];
}
}

__global__ void Im2Col_half(const ConvolutionCommon::Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const int maxCount,
    const int iBlock,
    const half* A,
    half* AP,
    const int ePack,
    const int eShift,
    DivModFast d_eA,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_fxy,
    DivModFast d_fx
    ) {
int eAlign = matmulParam->elhPack[0] << eShift;
int lAlign = matmulParam->elhPack[1];
int kernelCount = param->kernelX * param->kernelY;
for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
    size_t index = indexO >> 1;
    size_t lR = indexO & 1;
    int eIndex, lIndex;
    d_eA.divmod(index, lIndex, eIndex);
    size_t eU = eIndex >> eShift;
    size_t eR = eIndex & (ePack-1);
    size_t dstOffset = ((((eU * matmulParam->elhPack[1] + lIndex) << eShift) + eR) << 4) + (lR << 3);
    int4* dst = (int4*)(AP + dstOffset);

    eIndex += (iBlock*matmulParam->elhPack[0]*ePack);
    if (eIndex >= matmulParam->elh[0]) {
        *dst = {0, 0, 0, 0};
        continue;
    }
    // Compute for source
    int ox, oby, ob, oy, sz, kI, ksx, ksy;
    d_ow.divmod(eIndex, oby, ox);
    d_oh.divmod(oby, ob, oy);
    d_fxy.divmod(lIndex, sz, kI);
    d_fx.divmod(kI, ksy, ksx);

    size_t sx = ox * param->strideX + ksx * param->dilateX - param->padX;
    size_t sy = oy * param->strideY + ksy * param->dilateY - param->padY;
    if (sx >= 0 && sx < param->iw) {
        if (sy >=0 && sy < param->ih) {
            size_t offset = sz * param->srcZStep + (((ob * param->ih + sy) * param->iw + sx) << 4) + lR * 8;
            int4* src = (int4*)(A + offset);
            *dst = *src;
            continue;
        }
    }
    *dst = {0, 0, 0, 0};
}
}

__global__ void Im2Col_half_OPT(const ConvolutionCommon::Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const size_t maxCount,
    const int iBlock,
    const half* A,
    half* AP,
    const int ePack,
    const int eShift,
    DivModFast d_eA,
    DivModFast d_ow,
    DivModFast d_oh,
    DivModFast d_fxy,
    DivModFast d_fx
) {
size_t eAlign = matmulParam->elhPack[0] << eShift;
size_t lAlign = matmulParam->elhPack[1];
size_t kernelCount = param->kernelX * param->kernelY;
for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
    size_t index = indexO >> 2;
    size_t lR = indexO & 3;
    int eIndex, lIndex;
    d_eA.divmod(index, lIndex, eIndex);
    size_t eU = eIndex >> eShift;
    size_t eR = eIndex & (ePack-1);

    eIndex += (iBlock*matmulParam->elhPack[0]*ePack);

    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex) << eShift) + eR) << 4) + (lR << 2);

    int2* dst = (int2*)(AP + dstOffset);
    if (eIndex >= matmulParam->elh[0]) {
        *dst = {0, 0};
        continue;
    }

    // Compute for source
    int ox, oby, ob, oy, sz, kI, ksx, ksy;
    d_ow.divmod(eIndex, oby, ox);
    d_oh.divmod(oby, ob, oy);
    d_fxy.divmod(lIndex, sz, kI);
    d_fx.divmod(kI, ksy, ksx);

    size_t sx = ox * param->strideX + ksx * param->dilateX - param->padX;
    size_t sy = oy * param->strideY + ksy * param->dilateY - param->padY;

    if (sx >= 0 && sx < param->iw) {
        if (sy >=0 && sy < param->ih) {
            size_t offset = sz * param->srcZStep + (((ob * param->ih + sy) * param->iw + sx) << 4) + (lR << 2);
            int2* src = (int2*)(A + offset);
            *dst = *src;
            continue;
        }
    }
    *dst = {0, 0};
}
}


__global__ void Im2Col_half_3x3S1D1P1_OPT2(const ConvolutionCommon::Im2ColParameter* param,
    const MatMulParam* matmulParam,
    const size_t maxCount,
    const int iBlock,
    const half* A,
    half* AP,
    const int ePack,
    const int eShift,
    DivModFast d_eA,
    DivModFast d_ow,
    DivModFast d_oh
    ) {
for (size_t indexO = blockIdx.x * blockDim.x + threadIdx.x; indexO < maxCount; indexO += blockDim.x * gridDim.x) {
size_t index = indexO >> 3;
size_t lR = indexO & 7;
int eIndex, lIndex;
d_eA.divmod(index, lIndex, eIndex);

if (eIndex >= matmulParam->elh[0]) {
    continue;
}
int ix, oby, ob, iy;
d_ow.divmod(eIndex, oby, ix);
d_oh.divmod(oby, ob, iy);
size_t sz = lIndex;

size_t offset = sz * param->srcZStep + (((ob * param->ih + iy) * param->iw + ix) << 4) + (lR << 1);
int src = *((int*)(A + offset));

// Pixel (iy-1, ix-1)
if(iy-1 >=0 && ix-1 >=0) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy-1) * param->iw + (ix-1));
    size_t eU = oeIndex >> eShift;
    size_t eR = oeIndex & (ePack-1);
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 8) << eShift) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy-1 ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix-1 ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy-1, ix+0)
if(iy-1 >=0) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy-1) * param->iw + (ix+0));
    size_t eU = oeIndex >> eShift;
    size_t eR = oeIndex & (ePack-1);
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 7) << eShift) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy-1 ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy-1, ix+1)
if(iy-1 >=0 && ix+1 < param->iw) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy-1) * param->iw + (ix+1));
    size_t eU = oeIndex >> eShift;
    size_t eR = oeIndex & (ePack-1);
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 6) << eShift) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy-1 ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix+1 == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy+0, ix-1)
if(ix-1 >=0) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+0) * param->iw + (ix-1));
    size_t eU = oeIndex >> eShift;
    size_t eR = oeIndex & (ePack-1);
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 5) << eShift) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(iy == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix-1 ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy, ix)
if(1) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+0) * param->iw + (ix+0));
    size_t eU = oeIndex >> eShift;
    size_t eR = oeIndex & (ePack-1);
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 4) << eShift) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(iy == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy, ix+1)
if(ix+1 < param->iw) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+0) * param->iw + (ix+1));
    size_t eU = oeIndex >> eShift;
    size_t eR = oeIndex & (ePack-1);
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 3) << eShift) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy ==0) {
        size_t index[3] = {0, 1, 2};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(iy == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix+1 == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

// Pixel (iy+1, ix-1)
if(iy+1 < param->ih && ix-1 >=0) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+1) * param->iw + (ix-1));
    size_t eU = oeIndex >> eShift;
    size_t eR = oeIndex & (ePack-1);
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 2) << eShift) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy+1 == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix-1 ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }  
}

// Pixel (iy+1, ix)
if(iy+1 < param->ih) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+1) * param->iw + (ix+0));
    size_t eU = oeIndex >> eShift;
    size_t eR = oeIndex & (ePack-1);
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 1) << eShift) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy+1 == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix ==0) {
        size_t index[3] = {0, 3, 6};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}

//Pixel (iy+1, ix+1)
if(iy+1 < param->ih && ix+1 < param->iw) {
    size_t oeIndex = (ob * param->ih * param->iw + (iy+1) * param->iw + (ix+1));
    size_t eU = oeIndex >> eShift;
    size_t eR = oeIndex & (ePack-1);
    size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + 0) << eShift) + eR) << 4) + (lR << 1);
    int* dst = (int*)(AP + dstOffset);
    *dst = src;

    // Corner case
    if(iy+1 == param->ih-1) {
        size_t index[3] = {6, 7, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
    if(ix+1 == param->iw-1) {
        size_t index[3] = {2, 5, 8};
        for(size_t i=0; i<3; i++) {
            size_t dstOffset = ((((eU * (size_t)matmulParam->elhPack[1] + lIndex*9 + index[i]) << eShift) + eR) << 4) + (lR << 1);
            int* dst = (int*)(AP + dstOffset);
            *dst = 0;
        }
    }
}
}
}



void Im2ColMain(CUDARuntime* runtime, const MatMulParam* cpuMatlMul, const MatMulParam* gpuMatMul, const ConvolutionCommon::Im2ColParameter* cpuIm2Col, const ConvolutionCommon::Im2ColParameter* gpuIm2Col,\
     const Tensor* input, __half* mIm2ColBuffer, int ePack, int eShift, int bytes, int iBlock) {

    const void *input_addr = (const void*)input->deviceId();
    size_t eAlign = cpuMatlMul->elhPack[0] * ePack;
    size_t lAlign = cpuMatlMul->elhPack[1];

    DivModFast eAlignD(eAlign);
    DivModFast owD(cpuIm2Col->ow);
    DivModFast ohD(cpuIm2Col->oh);

    if (cpuIm2Col->kernelX == 1 && cpuIm2Col->kernelY == 1 && \
        cpuMatlMul->elh[0] % 16 == 0 && \
        cpuIm2Col->strideX == 1 && cpuIm2Col->strideY == 1 && \
        cpuIm2Col->dilateX == 1 && cpuIm2Col->dilateY == 1 && \
        cpuIm2Col->padX == 0 && cpuIm2Col->padY == 0) {

        size_t maxCount = eAlign * lAlign * 8;//Align 2
        int block_num = runtime->blocks_num(maxCount);
        int block_size = runtime->threads_num();
        if(bytes == 4) {
            Im2Col1x1_OPT<<<block_num, block_size>>>(gpuIm2Col, gpuMatMul, maxCount,
                    (const float*)input_addr, mIm2ColBuffer, ePack, eShift, eAlignD, owD, ohD);
            checkKernelErrors;
        } else {
            Im2Col1x1_half_OPT<<<block_num, block_size>>>(gpuIm2Col, gpuMatMul, maxCount,
                    (const half*)input_addr, mIm2ColBuffer, ePack, eShift, eAlignD, owD, ohD);
            checkKernelErrors;
        }
    } else if (cpuIm2Col->kernelX == 1 && cpuIm2Col->kernelY == 1) {
        size_t maxCount = eAlign * lAlign * 2;//Align 8
        int block_num = runtime->blocks_num(maxCount);
        int block_size = runtime->threads_num();
        if(bytes == 4) {
            Im2Col1x1<<<block_num, block_size>>>(gpuIm2Col, gpuMatMul, (const float*)input_addr, mIm2ColBuffer, ePack, eShift, eAlignD, owD, ohD);
            checkKernelErrors;
        } else {
            Im2Col1x1_half<<<block_num, block_size>>>(gpuIm2Col, gpuMatMul, (const half*)input_addr, mIm2ColBuffer, ePack, eShift, eAlignD, owD, ohD);
            checkKernelErrors;
        }
    } else if(eAlign == cpuMatlMul->elh[0] && iBlock == 0 && \
        cpuIm2Col->kernelX == 3 && cpuIm2Col->kernelY == 3 && \
        cpuMatlMul->elh[0] % 16 == 0 && \
        cpuIm2Col->strideX == 1 && cpuIm2Col->strideY == 1 && \
        cpuIm2Col->dilateX == 1 && cpuIm2Col->dilateY == 1 && \
        cpuIm2Col->padX == 1 && cpuIm2Col->padY == 1 && \
        bytes == 2) {
        
        size_t maxCount = eAlign * (lAlign / 9) * 8;
        size_t block_num = runtime->blocks_num(maxCount);
        size_t block_size = runtime->threads_num();

        // printf("%d: %d-%d-%d-%d-%d, %d-%d\n", iBlock, cpuIm2Col->icDiv4, cpuIm2Col->ih, cpuIm2Col->iw, cpuIm2Col->oh, cpuIm2Col->ow, eAlign, lAlign);
        Im2Col_half_3x3S1D1P1_OPT2<<<block_num, block_size>>>(gpuIm2Col, gpuMatMul, maxCount, iBlock, (const half*)input_addr, mIm2ColBuffer,\
            ePack, eShift, eAlignD, owD, ohD);
        checkKernelErrors;
    } else {
        size_t maxCount = eAlign * lAlign * 2;
        size_t block_num = runtime->blocks_num(maxCount);
        size_t block_size = runtime->threads_num();
        if(bytes == 4) {
            Im2Col<<<block_num, block_size>>>(gpuIm2Col, gpuMatMul, iBlock, (const float*)input_addr, mIm2ColBuffer, ePack, eShift);
            checkKernelErrors;
        } else {
            //printf("%d-%d-%d-%d-%d, %d-%d\n", cpuIm2Col->icDiv4, cpuIm2Col->ih, cpuIm2Col->iw, cpuIm2Col->oh, cpuIm2Col->ow, eAlign, lAlign);

            DivModFast fxyD((cpuIm2Col->kernelX*cpuIm2Col->kernelY));
            DivModFast fxD(cpuIm2Col->kernelX);
            maxCount = eAlign * lAlign * 4;
            block_num = runtime->blocks_num(maxCount);
            block_size = runtime->threads_num();

            //Im2Col_half<<<block_num, block_size>>>(gpuIm2Col, gpuMatMul, maxCount, (const half*)input_addr, mIm2ColBuffer, eAlignD, owD, ohD, fxyD, fxD);
            Im2Col_half_OPT<<<block_num, block_size>>>(gpuIm2Col, gpuMatMul, maxCount, iBlock, (const half*)input_addr, mIm2ColBuffer, \
                ePack, eShift, eAlignD, owD, ohD, fxyD, fxD);
            checkKernelErrors;
        }
    }
}

} // namespace CUDA
} // namespace MNN