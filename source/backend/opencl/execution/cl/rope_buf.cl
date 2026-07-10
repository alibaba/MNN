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
    if (x >= 1 || y >= outerSize || z >= fullHead) {
        return;
    }
#else
    if (x >= workDim || y >= outerSize || z >= fullHead) {
        return;
    }
#endif

    const int D = headDim;
    bool isQ = (z < numHead);
    __global const FLOAT* in_ptr = isQ ? q : k;
    const int inBase = isQ ? (z * D) : ((z - numHead) * D);
    __global FLOAT* out_ptr = isQ ? q_out : k_out;
    const int outBase = inBase;

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
    for (int i = 0; i < ropeHalfD; ++i) {
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
        out_ptr[C4_OFFSET(y, outBase + i, outerSize)] = v0;
        out_ptr[C4_OFFSET(y, outBase + i + ropeHalfD, outerSize)] = v1;
    }
    for (int i = 2 * ropeHalfD; i < D; ++i) {
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
        out_ptr[C4_OFFSET(y, outBase + i, outerSize)] = value;
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
        out_ptr[C4_OFFSET(y, outBase + x, outerSize)] = v0;
        out_ptr[C4_OFFSET(y, outBase + x + ropeHalfD, outerSize)] = v1;
    }
    int tail = 2 * ropeHalfD + x;
    if (tail < D) {
        out_ptr[C4_OFFSET(y, outBase + tail, outerSize)] = in_ptr[C4_OFFSET(y, inBase + tail, outerSize)];
    }
#endif
}
