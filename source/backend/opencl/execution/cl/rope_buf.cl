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

// Capacity of the norm-reduction scratch below. The host clamps normSplit to
// kRopeNormSplitMax (RopeBufExecution.cpp); keep it <= this value.
#define NORM_SPLIT_MAX 32

__kernel void rope_buf(GLOBAL_SIZE_3_DIMS __global const FLOAT* q, __global const FLOAT* k, __global const FLOAT* cos,
                       __global const FLOAT* sin, __global FLOAT* q_out, __global FLOAT* k_out,
                       __private const int outerSize, __private const int workDim, __private const int ropeHalfD,
                       __private const int headDim, __private const int numHead, __private const int kvNumHead
#ifdef Q_NORM
                       ,
                       __private const float qEps
#ifdef Q_GAMMA
                       ,
                       __global const float* qGamma
#endif
#endif
#ifdef K_NORM
                       ,
                       __private const float kEps
#ifdef K_GAMMA
                       ,
                       __global const float* kGamma
#endif
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

#if defined(Q_NORM) || defined(K_NORM)
#if defined(Q_NORM) && defined(K_NORM)
    const bool hasNorm = true;
    const float eps = isQ ? qEps : kEps;
#elif defined(Q_NORM)
    const bool hasNorm = isQ;
    const float eps = qEps;
#else
    const bool hasNorm = !isQ;
    const float eps = kEps;
#endif
    float var = 1.0f;
#if defined(NORM_SPLIT_REDUCE)
    // Each split sums its own slice of the head, one local-memory reduction
    // shares the total. Host pins LWS.x == split so all splits of one head
    // sit in the same workgroup, and z is workgroup-uniform, so isQ is too
    // and the barrier never diverges.
    __local float sNorm[NORM_SPLIT_MAX];
    if (hasNorm) {
        // Stride by split so the workgroup's lanes read consecutive channels
        // at each step (coalesced), instead of block-wise slices 16B apart.
        float part = 0.0f;
#ifdef INTERLEAVED
        if ((D & 3) == 0) {
            for (int qi = x; qi < (D >> 2); qi += split) {
                const float4 xv = convert_float4(vload4(0, in_ptr + (((inBase >> 2) + qi) * outerSize + y) * 4));
                part += dot(xv, xv);
            }
        } else
#endif
        {
            for (int i = x; i < D; i += split) {
                float val = (float)in_ptr[C4_OFFSET(y, inBase + i, outerSize)];
                part += val * val;
            }
        }
        sNorm[x] = part;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (hasNorm) {
        float total = 0.0f;
        for (int s = 0; s < split; ++s) {
            total += sNorm[s];
        }
        var = 1.0f / sqrt(total / D + eps);
    }
#else
    if (hasNorm) {
        float part = 0.0f;
#ifdef INTERLEAVED
        if ((D & 3) == 0) {
            for (int qi = 0; qi < (D >> 2); ++qi) {
                const float4 xv = convert_float4(vload4(0, in_ptr + (((inBase >> 2) + qi) * outerSize + y) * 4));
                part += dot(xv, xv);
            }
        } else
#endif
        {
            for (int i = 0; i < D; ++i) {
                float val = (float)in_ptr[C4_OFFSET(y, inBase + i, outerSize)];
                part += val * val;
            }
        }
        var = 1.0f / sqrt(part / D + eps);
    }
#endif
#endif

#if defined(Q_NORM) || defined(K_NORM)
#ifdef INTERLEAVED
    if ((D & 3) == 0 && (ropeHalfD & 1) == 0) {
        const int ropeQuads = ropeHalfD >> 1;
        const FLOAT4 sgn = (FLOAT4)((FLOAT)-1.0f, (FLOAT)1.0f, (FLOAT)-1.0f, (FLOAT)1.0f);
        for (int qi = x; qi < ropeQuads; qi += split) {
            const int ch = qi << 2;
            FLOAT4 xv = vload4(0, in_ptr + (((inBase >> 2) + qi) * outerSize + y) * 4);
            if (hasNorm) {
                float4 xf = convert_float4(xv) * var;
#if defined(Q_GAMMA) && defined(K_GAMMA)
                xf *= vload4(0, (isQ ? qGamma : kGamma) + ch);
#elif defined(Q_GAMMA)
                if (isQ) {
                    xf *= vload4(0, qGamma + ch);
                }
#elif defined(K_GAMMA)
                if (!isQ) {
                    xf *= vload4(0, kGamma + ch);
                }
#endif
                xv = CONVERT_FLOAT4(xf);
            }
            const int trig = y * (2 * ropeHalfD) + ch;
            const FLOAT4 cv = vload4(0, cos + trig);
            const FLOAT4 sv = vload4(0, sin + trig);
            vstore4(xv * cv + sgn * xv.yxwz * sv, 0, out_ptr + ch);
        }
    } else {
        for (int i = x; i < ropeHalfD; i += split) {
            const int ch = 2 * i;
            const int trig = y * (2 * ropeHalfD) + ch;
            FLOAT evenVal = in_ptr[C4_OFFSET(y, inBase + ch, outerSize)];
            FLOAT oddVal = in_ptr[C4_OFFSET(y, inBase + ch + 1, outerSize)];
            if (hasNorm) {
                float e = (float)evenVal * var;
                float o = (float)oddVal * var;
#if defined(Q_GAMMA) && defined(K_GAMMA)
                const __global float* gamma = isQ ? qGamma : kGamma;
                e *= gamma[ch];
                o *= gamma[ch + 1];
#elif defined(Q_GAMMA)
                if (isQ) {
                    e *= qGamma[ch];
                    o *= qGamma[ch + 1];
                }
#elif defined(K_GAMMA)
                if (!isQ) {
                    e *= kGamma[ch];
                    o *= kGamma[ch + 1];
                }
#endif
                evenVal = (FLOAT)e;
                oddVal = (FLOAT)o;
            }
            const FLOAT c = cos[trig];
            const FLOAT s = sin[trig];
            out_ptr[ch] = evenVal * c - oddVal * s;
            out_ptr[ch + 1] = evenVal * s + oddVal * c;
        }
    }
#else
    for (int i = x; i < ropeHalfD; i += split) {
        const int cosIndex = y * (2 * ropeHalfD) + i;
        FLOAT cEven = cos[cosIndex];
        FLOAT cOdd = cos[cosIndex + ropeHalfD];
        FLOAT sEven = sin[cosIndex];
        FLOAT sOdd = sin[cosIndex + ropeHalfD];

        FLOAT evenVal = in_ptr[C4_OFFSET(y, inBase + i, outerSize)];
        FLOAT oddVal = in_ptr[C4_OFFSET(y, inBase + i + ropeHalfD, outerSize)];
#ifdef Q_NORM
        if (isQ) {
            evenVal = (FLOAT)((float)evenVal * var);
            oddVal = (FLOAT)((float)oddVal * var);
#ifdef Q_GAMMA
            evenVal = (FLOAT)((float)evenVal * qGamma[i]);
            oddVal = (FLOAT)((float)oddVal * qGamma[i + ropeHalfD]);
#endif
        }
#endif
#ifdef K_NORM
        if (!isQ) {
            evenVal = (FLOAT)((float)evenVal * var);
            oddVal = (FLOAT)((float)oddVal * var);
#ifdef K_GAMMA
            evenVal = (FLOAT)((float)evenVal * kGamma[i]);
            oddVal = (FLOAT)((float)oddVal * kGamma[i + ropeHalfD]);
#endif
        }
#endif

        FLOAT v0 = evenVal * cEven - oddVal * sEven;
        FLOAT v1 = oddVal * cOdd + evenVal * sOdd;
        out_ptr[i] = v0;
        out_ptr[i + ropeHalfD] = v1;
    }
#endif
    for (int i = 2 * ropeHalfD + x; i < D; i += split) {
        FLOAT value = in_ptr[C4_OFFSET(y, inBase + i, outerSize)];
        if (hasNorm) {
            float v = (float)value * var;
#if defined(Q_GAMMA) && defined(K_GAMMA)
            v *= (isQ ? qGamma : kGamma)[i];
#elif defined(Q_GAMMA)
            if (isQ) {
                v *= qGamma[i];
            }
#elif defined(K_GAMMA)
            if (!isQ) {
                v *= kGamma[i];
            }
#endif
            value = (FLOAT)v;
        }
        out_ptr[i] = value;
    }
#else
    if (x < ropeHalfD) {
#ifdef INTERLEAVED
        const int ch = 2 * x;
        const int trig = y * (2 * ropeHalfD) + ch;
        const FLOAT c = cos[trig];
        const FLOAT s = sin[trig];
        const FLOAT evenVal = in_ptr[C4_OFFSET(y, inBase + ch, outerSize)];
        const FLOAT oddVal = in_ptr[C4_OFFSET(y, inBase + ch + 1, outerSize)];
        out_ptr[ch] = evenVal * c - oddVal * s;
        out_ptr[ch + 1] = evenVal * s + oddVal * c;
#else
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
#endif
    }
    int tail = 2 * ropeHalfD + x;
    if (tail < D) {
        out_ptr[tail] = in_ptr[C4_OFFSET(y, inBase + tail, outerSize)];
    }
#endif
}
