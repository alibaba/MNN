#ifndef WINOGRAD_TRANS_
#define WINOGRAD_TRANS_

#include "TensorCoreGemm.cuh"
#include <cuda.h>
#include <mma.h>

using namespace nvcuda;

namespace MNN {
namespace CUDA {

template<typename T>
__global__ void WinoInputTrans(const T* input,
    half* BtdB,
    const int unit,
    const int block,
    const int ci,
    const int ciDiv,
    const int batch,
    const int wLen,
    const int hLen,
    const int ePack,
    const int lPack,
    const int pad_x,
    const int pad_y,
    const int width,
    const int height
) {
    const int eU = (batch * wLen * hLen + 15) / 16;
    const int lU = ciDiv;
    const int eP = eU * ePack;
    const int lP = lU * lPack;
    const int maxCount =  eP * lP;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {
        const int eU_idx    = index  / (lP * ePack);
        const int tmp1      = index  % (lP * ePack);
        const int lU_idx    = tmp1  / (ePack * lPack);
        const int tmp2      = tmp1  % (ePack * lPack);
        const int ePack_idx = tmp2  / lPack;
        const int lPack_idx = tmp2  % lPack;

        const int e_idx     = eU_idx * ePack + ePack_idx;
        const int batch_idx = e_idx  / (wLen * hLen);
        const int tmp3      = e_idx  % (wLen * hLen);
        const int w_idx     = tmp3   % wLen;
        const int h_idx     = tmp3   / wLen;

        const int sxStart = w_idx * unit - pad_x;
        const int syStart = h_idx * unit - pad_y;

        T S00, S10, S20, S30, S01, S11, S21, S31, S02, S12, S22, S32, S03, S13, S23, S33;
        
        int ci_p8 = ((ci + 7) / 8) * 8;
        int ci_idx = lU_idx * lPack + lPack_idx;
        int inp_offset = ((batch_idx * height + syStart) * width + sxStart) * ci_p8 + ci_idx;
        //(((lU_idx * batch + batch_idx) * height + syStart) * width + sxStart) * 16 + lPack_idx;
        {
            int sx      = 0 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S00         = outBound ? (T)(0) : input[inp_offset];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S10         = outBound ? (T)(0) : input[inp_offset+ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S20         = outBound ? (T)(0) : input[inp_offset+2*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 0 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S30         = outBound ? (T)(0) : input[inp_offset+3*ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S01         = outBound ? (T)(0) : input[inp_offset+width*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S11         = outBound ? (T)(0) : input[inp_offset+(width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S21         = outBound ? (T)(0) : input[inp_offset+(width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S31         = outBound ? (T)(0) : input[inp_offset+(width+3)*ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S02         = outBound ? (T)(0) : input[inp_offset+(2*width+0)*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S12         = outBound ? (T)(0) : input[inp_offset+(2*width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S22         = outBound ? (T)(0) : input[inp_offset+(2*width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S32         = outBound ? (T)(0) : input[inp_offset+(2*width+3)*ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S03         = outBound ? (T)(0) : input[inp_offset+(3*width+0)*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S13         = outBound ? (T)(0) : input[inp_offset+(3*width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S23         = outBound ? (T)(0) : input[inp_offset+(3*width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S33         = outBound ? (T)(0) : input[inp_offset+(3*width+3)*ci_p8];
        }
        T m00 = +S00 - S02;
        T m10 = +S10 - S12;
        T m20 = +S20 - S22;
        T m30 = +S30 - S32;
        T m01 = +(T)0.5f * S01 + (T)0.5f * S02;
        T m11 = +(T)0.5f * S11 + (T)0.5f * S12;
        T m21 = +(T)0.5f * S21 + (T)0.5f * S22;
        T m31 = +(T)0.5f * S31 + (T)0.5f * S32;
        T m02 = -(T)0.5f * S01 + (T)0.5f * S02;
        T m12 = -(T)0.5f * S11 + (T)0.5f * S12;
        T m22 = -(T)0.5f * S21 + (T)0.5f * S22;
        T m32 = -(T)0.5f * S31 + (T)0.5f * S32;
        T m03 = -S01 + S03;
        T m13 = -S11 + S13;
        T m23 = -S21 + S23;
        T m33 = -S31 + S33;

        BtdB[0*maxCount + index]  = +m00 - m20;
        BtdB[1*maxCount + index]  = +(T)0.5f * m10 + (T)0.5f * m20;
        BtdB[2*maxCount + index]  = -(T)0.5f * m10 + (T)0.5f * m20;
        BtdB[3*maxCount + index]  = -m10 + m30;
        BtdB[4*maxCount + index]  = +m01 - m21;
        BtdB[5*maxCount + index]  = +(T)0.5f * m11 + (T)0.5f * m21;
        BtdB[6*maxCount + index]  = -(T)0.5f * m11 + (T)0.5f * m21;
        BtdB[7*maxCount + index]  = -m11 + m31;
        BtdB[8*maxCount + index]  = +m02 - m22;
        BtdB[9*maxCount + index]  = +(T)0.5f * m12 + (T)0.5f * m22;
        BtdB[10*maxCount + index] = -(T)0.5f * m12 + (T)0.5f * m22;
        BtdB[11*maxCount + index] = -m12 + m32;
        BtdB[12*maxCount + index] = +m03 - m23;
        BtdB[13*maxCount + index] = +(T)0.5f * m13 + (T)0.5f * m23;
        BtdB[14*maxCount + index] = -(T)0.5f * m13 + (T)0.5f * m23;
        BtdB[15*maxCount + index] = -m13 + m33;
    }
}


template<typename T>
__global__ void WinoTrans2Output(const T* matmulData,
    const float* biasData,
    T* output,
    const MatMulParam* param,
    const int unit,
    const int block,
    const int co,
    const int ciDiv,
    const int batch,
    const int wLen,
    const int hLen,
    const int ePack,
    const int hPack,
    const int width,
    const int height
) {
    size_t e = param->elh[0];
    size_t hU = param->elhPack[2];
    const size_t maxCount =  e * hU * hPack;

    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {

        const int hU_idx    = index  / (e * hPack);
        const int tmp1      = index  % (e * hPack);
        const int e_idx    = tmp1  / hPack;
        const int hPack_idx      = tmp1  % hPack;

        const int batch_idx = e_idx  / (wLen * hLen);
        const int tmp3      = e_idx  % (wLen * hLen);
        const int w_idx     = tmp3   % wLen;
        const int h_idx     = tmp3   / wLen;
        int basic_offset = (hU_idx * e + e_idx) * hPack + hPack_idx;

        auto S00 = matmulData[basic_offset];
        auto S10 = matmulData[basic_offset + maxCount];
        auto S20 = matmulData[basic_offset + maxCount * 2];
        auto S30 = matmulData[basic_offset + maxCount * 3];
        auto S01 = matmulData[basic_offset + maxCount * 4];
        auto S11 = matmulData[basic_offset + maxCount * 5];
        auto S21 = matmulData[basic_offset + maxCount * 6];
        auto S31 = matmulData[basic_offset + maxCount * 7];
        auto S02 = matmulData[basic_offset + maxCount * 8];
        auto S12 = matmulData[basic_offset + maxCount * 9];
        auto S22 = matmulData[basic_offset + maxCount * 10];
        auto S32 = matmulData[basic_offset + maxCount * 11];
        auto S03 = matmulData[basic_offset + maxCount * 12];
        auto S13 = matmulData[basic_offset + maxCount * 13];
        auto S23 = matmulData[basic_offset + maxCount * 14];
        auto S33 = matmulData[basic_offset + maxCount * 15];

        auto m00 = +S00 + S01 + S02;
        auto m10 = +S10 + S11 + S12;
        auto m20 = +S20 + S21 + S22;
        auto m30 = +S30 + S31 + S32;
        auto m01 = +S01 - S02 + S03;
        auto m11 = +S11 - S12 + S13;
        auto m21 = +S21 - S22 + S23;
        auto m31 = +S31 - S32 + S33;

        // write output
        float bias = biasData[hU_idx * hPack + hPack_idx];

        const int dxStart = w_idx * unit;
        const int dyStart = h_idx * unit;

        int co_p8 = ((co + 7) / 8) * 8;
        int co_idx = hU_idx * hPack + hPack_idx;
        if(co_idx >= co_p8) {
            continue;
        }
        int out_offset = ((batch_idx * height + dyStart) * width + dxStart) * co_p8 + co_idx;
        //(((hU_idx * batch + batch_idx) * height + dyStart) * width + dxStart) * 16 + hPack_idx;

        /* if true */ {
            float res = bias + (float)(m00 + m10 + m20);
            res = max(res, param->minValue);
            res = min(res, param->maxValue);
            output[out_offset] = (T)res;
        }
        if (dxStart + 1 < width) {
            float res = bias + (float)(m10 - m20 + m30);
            res = max(res, param->minValue);
            res = min(res, param->maxValue);
            output[out_offset + co_p8] = (T)res;
        }
        if (dyStart + 1 < height) {
            float res = bias + (float)(m01 + m11 + m21);
            res = max(res, param->minValue);
            res = min(res, param->maxValue);
            output[out_offset + width * co_p8] = (T)res;
        }
        if (dxStart + 1 < width && dyStart + 1 < height) {
            float res = bias + (float)(m11 - m21 + m31);
            res = max(res, param->minValue);
            res = min(res, param->maxValue);
            output[out_offset + (width + 1) * co_p8] = (T)res;
        }

    }
}

template<typename T>
__global__ void GemmPackedMulti(const MatMulParam* param, const int iBlock, const int multi_num,
    T *c, const half *a, const half *b) {

    size_t eU = param->elhPack[0];
    size_t lU = param->elhPack[1];
    size_t hU = param->elhPack[2];
    size_t maxCount = multi_num * eU * hU * warpSize;
    size_t wrapId = threadIdx.x / warpSize;
    size_t laneId = threadIdx.x % warpSize;
    extern __shared__ float sharedMemory[];

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        size_t tmp = index / warpSize;
        size_t subIndex = tmp % (eU * hU);
        size_t blockId = tmp / (eU * hU);
        size_t warpM = subIndex % eU;
        size_t warpN = subIndex / eU;
        T* cache = (T*)(sharedMemory + wrapId * 16 * 16);
        // Declare the fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, T> acc_frag;

        wmma::fill_fragment(acc_frag, (T)0.0f);
        //wmma::load_matrix_sync(acc_frag, biasPtr + 16 * warpN, 0, wmma::mem_row_major);
        const half* aStart = a + (blockId * eU + warpM) * lU * 16 * 16;
        const half* bStart = b + (blockId * hU + warpN) * lU * 16 * 16;
        //printf("GemmPacked: %d - %d - %d, numele: %d, %d\n", eU, lU, hU, a_frag.num_elements, b_frag.num_elements);
        // MLA
        for (size_t i = 0; i < lU; ++i) {
            wmma::load_matrix_sync(a_frag, aStart + i * 256, 16);
            wmma::load_matrix_sync(b_frag, bStart + i * 256, 16);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        size_t eSta = (warpM + iBlock*eU) * 16;
        if(eSta >= (size_t)param->elh[0]) {
            continue;
        }
        size_t eEnd = ((eSta + (size_t)16) > (size_t)param->elh[0]) ? (size_t)param->elh[0] : (eSta + (size_t)16);

        size_t eC = eEnd - eSta;
        T* dstStart = (T*)(c + (blockId * hU + warpN) * 16 * (size_t)param->elh[0] + eSta * 16);
        wmma::store_matrix_sync(cache, acc_frag, 16, wmma::mem_row_major);
        if (warpSize % 16 == 0) {
            if(sizeof(T) == 4) {
                size_t r = warpSize / 16;
                size_t x = laneId / r;
                size_t ysta = laneId % r;
                for (size_t y = ysta; y < eC; y+=r) {
                    float value = *((T*)(cache + 16 * y + x));
                    dstStart[y * 16 + x] = value;
                }
            } else {
                size_t xsta = (laneId % 8) * 2;
                size_t ysta = laneId / 8;
                for (size_t y = ysta; y < eC; y+=4) {
                    dstStart[y * 16 + xsta]     = *((T*)(cache + 16 * y + xsta));
                    dstStart[y * 16 + xsta + 1] = *((T*)(cache + 16 * y + xsta + 1));
                }
            }
        } else {
            for (size_t tId = laneId; tId < eC * 16; tId += warpSize) {
                size_t y = tId % eC;
                size_t x = tId / eC;
                float value = *((T*)(cache + 16 * y + x));
                dstStart[y * 16 + x] = value;
            }
        }
    }
}

} //namespace CUDA
} //namespace MNN
#endif