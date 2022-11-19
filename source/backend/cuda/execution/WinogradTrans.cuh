#ifndef WINOGRAD_TRANS_
#define WINOGRAD_TRANS_

namespace MNN {
namespace CUDA {

template<typename T0, typename T1>
__global__ void WinoInputTrans(const T0* input,
    T1* BtdB,
    const int unit,
    const int block,
    const int ci,
    const int ci_p8,
    const int maxCount,
    DivModFast lD,
    DivModFast whD,
    DivModFast wD,
    const int pad_x,
    const int pad_y,
    const int width,
    const int height
) {
    const int l = ci_p8;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {
        int e_idx, ci_idx, batch_idx, tmp, w_idx, h_idx;
        lD.divmod(index, e_idx, ci_idx);
        whD.divmod(e_idx, batch_idx, tmp);
        wD.divmod(tmp, h_idx, w_idx);

        const int sxStart = w_idx * unit - pad_x;
        const int syStart = h_idx * unit - pad_y;

        T0 S00, S10, S20, S30, S01, S11, S21, S31, S02, S12, S22, S32, S03, S13, S23, S33;
        
        int inp_offset = ((batch_idx * height + syStart) * width + sxStart) * ci_p8 + ci_idx;
        {
            int sx      = 0 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S00         = outBound ? (T0)(0) : input[inp_offset];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S10         = outBound ? (T0)(0) : input[inp_offset+ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S20         = outBound ? (T0)(0) : input[inp_offset+ci_p8+ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 0 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S30         = outBound ? (T0)(0) : input[inp_offset+ci_p8+ci_p8+ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S01         = outBound ? (T0)(0) : input[inp_offset+width*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S11         = outBound ? (T0)(0) : input[inp_offset+(width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S21         = outBound ? (T0)(0) : input[inp_offset+(width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S31         = outBound ? (T0)(0) : input[inp_offset+(width+3)*ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S02         = outBound ? (T0)(0) : input[inp_offset+(width+width+0)*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S12         = outBound ? (T0)(0) : input[inp_offset+(width+width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S22         = outBound ? (T0)(0) : input[inp_offset+(width+width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S32         = outBound ? (T0)(0) : input[inp_offset+(width+width+3)*ci_p8];
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S03         = outBound ? (T0)(0) : input[inp_offset+(width+width+width+0)*ci_p8];
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S13         = outBound ? (T0)(0) : input[inp_offset+(width+width+width+1)*ci_p8];
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S23         = outBound ? (T0)(0) : input[inp_offset+(width+width+width+2)*ci_p8];
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 3 + syStart;
            bool outBound = (sx < 0 || sx >= width || sy < 0 || sy >= height || ci_idx >= ci);
            S33         = outBound ? (T0)(0) : input[inp_offset+(width+width+width+3)*ci_p8];
        }
        T0 m00 = +S00 - S02;
        T0 m10 = +S10 - S12;
        T0 m20 = +S20 - S22;
        T0 m30 = +S30 - S32;
        T0 m01 = +(T0)0.5f * (S01 + S02);
        T0 m11 = +(T0)0.5f * (S11 + S12);
        T0 m21 = +(T0)0.5f * (S21 + S22);
        T0 m31 = +(T0)0.5f * (S31 + S32);
        T0 m02 = +(T0)0.5f * (-S01 + S02);
        T0 m12 = +(T0)0.5f * (-S11 + S12);
        T0 m22 = +(T0)0.5f * (-S21 + S22);
        T0 m32 = +(T0)0.5f * (-S31 + S32);
        T0 m03 = -S01 + S03;
        T0 m13 = -S11 + S13;
        T0 m23 = -S21 + S23;
        T0 m33 = -S31 + S33;

        BtdB[0*maxCount + index]  = +m00 - m20;
        BtdB[1*maxCount + index]  = +(T0)0.5f * (m10 + m20);
        BtdB[2*maxCount + index]  = +(T0)0.5f * (-m10 + m20);
        BtdB[3*maxCount + index]  = -m10 + m30;
        BtdB[4*maxCount + index]  = +m01 - m21;
        BtdB[5*maxCount + index]  = +(T0)0.5f * (m11 + m21);
        BtdB[6*maxCount + index]  = +(T0)0.5f * (-m11 + m21);
        BtdB[7*maxCount + index]  = -m11 + m31;
        BtdB[8*maxCount + index]  = +m02 - m22;
        BtdB[9*maxCount + index]  = +(T0)0.5f * (m12 + m22);
        BtdB[10*maxCount + index] = +(T0)0.5f * (-m12 + m22);
        BtdB[11*maxCount + index] = -m12 + m32;
        BtdB[12*maxCount + index] = +m03 - m23;
        BtdB[13*maxCount + index] = +(T0)0.5f * (m13 + m23);
        BtdB[14*maxCount + index] = +(T0)0.5f * (-m13 + m23);
        BtdB[15*maxCount + index] = -m13 + m33;
    }
}


template<typename T>
__global__ void WinoTrans2Output(const T* matmulData,
    const float* biasData,
    T* output,
    const int unit,
    const int block,
    const int co,
    const int co_p8,
    const int maxCount,
    DivModFast hD,
    DivModFast whD,
    DivModFast wD,
    const int width,
    const int height,
    const int activationType
) {
    const int h = co_p8;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {

        int e_idx, co_idx, batch_idx, tmp, w_idx, h_idx;
        hD.divmod(index, e_idx, co_idx);
        whD.divmod(e_idx, batch_idx, tmp);
        wD.divmod(tmp, h_idx, w_idx);

        auto S00 = matmulData[index];
        auto S10 = matmulData[index + maxCount];
        auto S20 = matmulData[index + maxCount * 2];
        auto S30 = matmulData[index + maxCount * 3];
        auto S01 = matmulData[index + maxCount * 4];
        auto S11 = matmulData[index + maxCount * 5];
        auto S21 = matmulData[index + maxCount * 6];
        auto S31 = matmulData[index + maxCount * 7];
        auto S02 = matmulData[index + maxCount * 8];
        auto S12 = matmulData[index + maxCount * 9];
        auto S22 = matmulData[index + maxCount * 10];
        auto S32 = matmulData[index + maxCount * 11];
        auto S03 = matmulData[index + maxCount * 12];
        auto S13 = matmulData[index + maxCount * 13];
        auto S23 = matmulData[index + maxCount * 14];
        auto S33 = matmulData[index + maxCount * 15];

        auto m00 = +S00 + S01 + S02;
        auto m10 = +S10 + S11 + S12;
        auto m20 = +S20 + S21 + S22;
        auto m30 = +S30 + S31 + S32;
        auto m01 = +S01 - S02 + S03;
        auto m11 = +S11 - S12 + S13;
        auto m21 = +S21 - S22 + S23;
        auto m31 = +S31 - S32 + S33;

        // write output
        float bias = biasData[co_idx];

        const int dxStart = w_idx * unit;
        const int dyStart = h_idx * unit;

        if(co_idx >= co_p8) {
            continue;
        }
        int out_offset = ((batch_idx * height + dyStart) * width + dxStart) * co_p8 + co_idx;

        /* if true */ {
            float res = bias + (float)(m00 + m10 + m20);
            if(activationType == 1) {
                res = max(res, 0.0f);
            }
            if(activationType == 2) {
                res = max(res, 0.0f);
                res = min(res, 6.0f);
            }
            output[out_offset] = (T)res;
        }
        if (dxStart + 1 < width) {
            float res = bias + (float)(m10 - m20 + m30);
            if(activationType == 1) {
                res = max(res, 0.0f);
            }
            if(activationType == 2) {
                res = max(res, 0.0f);
                res = min(res, 6.0f);
            }
            output[out_offset + co_p8] = (T)res;
        }
        if (dyStart + 1 < height) {
            float res = bias + (float)(m01 + m11 + m21);
            if(activationType == 1) {
                res = max(res, 0.0f);
            }
            if(activationType == 2) {
                res = max(res, 0.0f);
                res = min(res, 6.0f);
            }
            output[out_offset + width * co_p8] = (T)res;
        }
        if (dxStart + 1 < width && dyStart + 1 < height) {
            float res = bias + (float)(m11 - m21 + m31);
            if(activationType == 1) {
                res = max(res, 0.0f);
            }
            if(activationType == 2) {
                res = max(res, 0.0f);
                res = min(res, 6.0f);
            }
            output[out_offset + (width + 1) * co_p8] = (T)res;
        }

    }
}

} //namespace CUDA
} //namespace MNN
#endif