#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void rope_buf(GLOBAL_SIZE_3_DIMS
                       __global const FLOAT *q,
                       __global const FLOAT *k,
                       __global const FLOAT *cosEven,
                       __global const FLOAT *cosOdd,
                       __global const FLOAT *sinEven,
                       __global const FLOAT *sinOdd,
                       __global FLOAT *q_out,
                       __global FLOAT *k_out,
                       __private const int outerSize,
                       __private const int halfD,
                       __private const int ropeHalfD,
                       __private const int headDim,
                       __private const int numHead,
                       __private const int kvNumHead
#ifdef Q_NORM
                       , __global const float *qGamma
                       , __private const float qEps
#endif
#ifdef K_NORM
                       , __global const float *kGamma
                       , __private const float kEps
#endif
                       ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int fullHead = numHead + kvNumHead;
#if defined(Q_NORM) || defined(K_NORM)
    if (x >= 1 || y >= outerSize || z >= fullHead) {
        return;
    }
#else
    if (x >= halfD || y >= outerSize || z >= fullHead) {
        return;
    }
#endif

    const int D = headDim;
    bool isQ = (z < numHead);
    __global const FLOAT *in_ptr = isQ ? (q + (y * numHead + z) * D) : (k + (y * kvNumHead + z - numHead) * D);
    __global FLOAT *out_ptr = isQ ? (q_out + (y * numHead + z) * D) : (k_out + (y * kvNumHead + z - numHead) * D);

    float var = 0.0f;
#ifdef Q_NORM
    if (isQ) {
        for (int i = 0; i < D; ++i) {
            float val = (float)in_ptr[i];
            var += val * val;
        }
        var = 1.0f / sqrt(var / D + qEps);
    }
#endif
#ifdef K_NORM
    if (!isQ) {
        for (int i = 0; i < D; ++i) {
            float val = (float)in_ptr[i];
            var += val * val;
        }
        var = 1.0f / sqrt(var / D + kEps);
    }
#endif

#if defined(Q_NORM) || defined(K_NORM)
    for (int i = 0; i < halfD; ++i) {
        const int cosIndex = y * halfD + i;
        FLOAT cEven = cosEven[cosIndex];
        FLOAT cOdd = cosOdd[cosIndex];
        FLOAT sEven = sinEven[cosIndex];
        FLOAT sOdd = sinOdd[cosIndex];

        FLOAT evenVal = in_ptr[i];
        FLOAT oddVal = in_ptr[i + halfD];
#ifdef Q_NORM
        if (isQ) {
            evenVal = (FLOAT)((float)evenVal * var * qGamma[i]);
            oddVal = (FLOAT)((float)oddVal * var * qGamma[i + halfD]);
        }
#endif
#ifdef K_NORM
        if (!isQ) {
            evenVal = (FLOAT)((float)evenVal * var * kGamma[i]);
            oddVal = (FLOAT)((float)oddVal * var * kGamma[i + halfD]);
        }
#endif

        if (i < ropeHalfD) {
            FLOAT v0 = evenVal * cEven - oddVal * sEven;
            FLOAT v1 = oddVal * cOdd + evenVal * sOdd;
            out_ptr[i] = v0;
            out_ptr[i + halfD] = v1;
        } else {
            out_ptr[i] = evenVal;
            out_ptr[i + halfD] = oddVal;
        }
    }
#else
    const int cosIndex = y * halfD + x;
    FLOAT cEven = cosEven[cosIndex];
    FLOAT cOdd = cosOdd[cosIndex];
    FLOAT sEven = sinEven[cosIndex];
    FLOAT sOdd = sinOdd[cosIndex];

    FLOAT evenVal = in_ptr[x];
    FLOAT oddVal = in_ptr[x + halfD];

    if (x < ropeHalfD) {
        FLOAT v0 = evenVal * cEven - oddVal * sEven;
        FLOAT v1 = oddVal * cOdd + evenVal * sOdd;
        out_ptr[x] = v0;
        out_ptr[x + halfD] = v1;
    } else {
        out_ptr[x] = evenVal;
        out_ptr[x + halfD] = oddVal;
    }
#endif
}
