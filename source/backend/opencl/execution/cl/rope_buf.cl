#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define C4_OFFSET(token, channel, seqLen) (((channel) / 4) * (seqLen) * 4 + (token) * 4 + ((channel) % 4))

__kernel void rope_buf(GLOBAL_SIZE_3_DIMS __global const FLOAT* q, __global const FLOAT* k, __global const FLOAT* cos,
                       __global const FLOAT* sin, __global FLOAT* q_out, __global FLOAT* k_out,
                       __private const int outerSize, __private const int workDim, __private const int ropeHalfD,
                       __private const int headDim, __private const int numHead, __private const int kvNumHead
#ifdef Q_NORM
                       ,
                       __global const float* qGamma, __private const float qEps
#endif
#ifdef K_NORM
                       ,
                       __global const float* kGamma, __private const float kEps
#endif
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, y, z);

    const int fullHead = numHead + kvNumHead;
#if defined(Q_NORM) || defined(K_NORM)
    if (y >= outerSize || z >= fullHead) {
        return;
    }
    // The host splits the head dimension across `split` work-items so decode
    // (outerSize==1, fullHead~24) is not stuck at ~24 lanes; the RMS-norm pass
    // below is redundantly computed per split but stays in L1. split==1
    // reproduces the old single-lane behaviour exactly.
    const int split = global_size_dim0;
    const int chunk0 = x * ropeHalfD / split;
    const int chunk1 = (x + 1) * ropeHalfD / split;
#else
    if (x >= workDim || y >= outerSize || z >= fullHead) {
        return;
    }
#endif

    const int D = headDim;
    bool isQ = (z < numHead);
    __global const FLOAT* in_ptr = isQ ? q : k;
    const int inBase = isQ ? (z * D) : ((z - numHead) * D);
    __global FLOAT* out_ptr = isQ ? (q_out + (y * numHead + z) * D) :
                                      (k_out + (y * kvNumHead + z - numHead) * D);

    float var = 0.0f;
#ifdef Q_NORM
    if (isQ) {
        for (int i = 0; i < D; ++i) {
            float val = (float)in_ptr[C4_OFFSET(y, inBase + i, outerSize)];
            var += val * val;
        }
        var = 1.0f / sqrt(var / D + qEps);
    }
#endif
#ifdef K_NORM
    if (!isQ) {
        for (int i = 0; i < D; ++i) {
            float val = (float)in_ptr[C4_OFFSET(y, inBase + i, outerSize)];
            var += val * val;
        }
        var = 1.0f / sqrt(var / D + kEps);
    }
#endif

#if defined(Q_NORM) || defined(K_NORM)
    for (int i = chunk0; i < chunk1; ++i) {
        const int cosIndex = y * (2 * ropeHalfD) + i;
        FLOAT cEven = cos[cosIndex];
        FLOAT cOdd = cos[cosIndex + ropeHalfD];
        FLOAT sEven = sin[cosIndex];
        FLOAT sOdd = sin[cosIndex + ropeHalfD];

        FLOAT evenVal = in_ptr[C4_OFFSET(y, inBase + i, outerSize)];
        FLOAT oddVal = in_ptr[C4_OFFSET(y, inBase + i + ropeHalfD, outerSize)];
#ifdef Q_NORM
        if (isQ) {
            evenVal = (FLOAT)((float)evenVal * var * qGamma[i]);
            oddVal = (FLOAT)((float)oddVal * var * qGamma[i + ropeHalfD]);
        }
#endif
#ifdef K_NORM
        if (!isQ) {
            evenVal = (FLOAT)((float)evenVal * var * kGamma[i]);
            oddVal = (FLOAT)((float)oddVal * var * kGamma[i + ropeHalfD]);
        }
#endif

        FLOAT v0 = evenVal * cEven - oddVal * sEven;
        FLOAT v1 = oddVal * cOdd + evenVal * sOdd;
        out_ptr[i] = v0;
        out_ptr[i + ropeHalfD] = v1;
    }
    const int tailN = D - 2 * ropeHalfD;
    for (int t = x * tailN / split; t < (x + 1) * tailN / split; ++t) {
        const int i = 2 * ropeHalfD + t;
        FLOAT value = in_ptr[C4_OFFSET(y, inBase + i, outerSize)];
#ifdef Q_NORM
        if (isQ) {
            value = (FLOAT)((float)value * var * qGamma[i]);
        }
#endif
#ifdef K_NORM
        if (!isQ) {
            value = (FLOAT)((float)value * var * kGamma[i]);
        }
#endif
        out_ptr[i] = value;
    }
#else
    if (x < ropeHalfD) {
        const int cosIndex = y * (2 * ropeHalfD) + x;
        FLOAT cEven = cos[cosIndex];
        FLOAT cOdd = cos[cosIndex + ropeHalfD];
        FLOAT sEven = sin[cosIndex];
        FLOAT sOdd = sin[cosIndex + ropeHalfD];
        FLOAT evenVal = in_ptr[C4_OFFSET(y, inBase + x, outerSize)];
        FLOAT oddVal = in_ptr[C4_OFFSET(y, inBase + x + ropeHalfD, outerSize)];
        FLOAT v0 = evenVal * cEven - oddVal * sEven;
        FLOAT v1 = oddVal * cOdd + evenVal * sOdd;
        out_ptr[x] = v0;
        out_ptr[x + ropeHalfD] = v1;
    }
    int tail = 2 * ropeHalfD + x;
    if (tail < D) {
        out_ptr[tail] = in_ptr[C4_OFFSET(y, inBase + tail, outerSize)];
    }
#endif
}
