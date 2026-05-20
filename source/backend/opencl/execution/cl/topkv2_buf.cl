#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define THREAD_NUMBER 128
#define LOCAL_K 8
#define CANDIDATE_NUMBER (THREAD_NUMBER * LOCAL_K)

#ifdef IS_INT
typedef int DTYPE;
#else
typedef FLOAT DTYPE;
#endif

inline bool afterAsc(DTYPE aValue, int aIndex, DTYPE bValue, int bIndex) {
    if (aValue > bValue) {
        return true;
    }
    if (aValue < bValue) {
        return false;
    }
    return aIndex > bIndex;
}

inline bool better(DTYPE aValue, int aIndex, DTYPE bValue, int bIndex) {
    if (bIndex < 0) {
        return true;
    }
    if (aIndex < 0) {
        return false;
    }
#ifdef SORT_DESC
    if (aValue > bValue) {
        return true;
    }
    if (aValue < bValue) {
        return false;
    }
#else
    if (aValue < bValue) {
        return true;
    }
    if (aValue > bValue) {
        return false;
    }
#endif
    return aIndex < bIndex;
}

__kernel void topkv2_buf(GLOBAL_SIZE_3_DIMS
                       __global DTYPE *outValue,
                       __global int *outIndex,
                       __global const DTYPE *inValue,
                       __private const int rowSize,
                       __private const int k,
                       __private const int numRows) {
    const int gid0 = get_global_id(0);
    const int gid1 = get_global_id(1);
    const int gid2 = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(gid0, gid1, gid2);
    
    const int tid = get_local_id(0);
    const int row = get_group_id(1);

    if (tid >= THREAD_NUMBER || row >= numRows) {
        return;
    }

#ifdef IS_INT
    const DTYPE initWorst = (DTYPE)(2147483647);
    const DTYPE initBestWorst = (DTYPE)(-2147483648);
#else
    const DTYPE initWorst = (DTYPE)(FLT_MAX);
    const DTYPE initBestWorst = (DTYPE)(-FLT_MAX);
#endif

    DTYPE localValue[LOCAL_K];
    int localIndex[LOCAL_K];
#ifdef SORT_DESC
    for (uint i = 0; i < LOCAL_K; ++i) {
        localValue[i] = initBestWorst;
        localIndex[i] = -1;
    }
#else
    for (uint i = 0; i < LOCAL_K; ++i) {
        localValue[i] = initWorst;
        localIndex[i] = -1;
    }
#endif

    const __global DTYPE *rowIn = inValue + row * rowSize;

    for (int i = tid; i < rowSize; i += THREAD_NUMBER) {
        const DTYPE value = rowIn[i];
        if (!better(value, i, localValue[LOCAL_K - 1], localIndex[LOCAL_K - 1])) {
            continue;
        }

        uint insertPos = LOCAL_K;
        for (uint j = 0; j < LOCAL_K; ++j) {
            if (better(value, i, localValue[j], localIndex[j])) {
                insertPos = j;
                break;
            }
        }
        if (insertPos >= LOCAL_K) {
            continue;
        }
        for (uint j = LOCAL_K - 1; j > insertPos; --j) {
            localValue[j] = localValue[j - 1];
            localIndex[j] = localIndex[j - 1];
        }
        localValue[insertPos] = value;
        localIndex[insertPos] = i;
    }

    __local DTYPE sharedValue[CANDIDATE_NUMBER];
    __local int sharedIndex[CANDIDATE_NUMBER];
    const uint base = tid * LOCAL_K;
    for (uint i = 0; i < LOCAL_K; ++i) {
        sharedValue[base + i] = localValue[i];
        sharedIndex[base + i] = localIndex[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint size = 2; size <= CANDIDATE_NUMBER; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint idx = tid; idx < CANDIDATE_NUMBER; idx += THREAD_NUMBER) {
                const uint ixj = idx ^ stride;
                if (ixj <= idx) {
                    continue;
                }
                bool up = ((idx & size) == 0);
#ifdef SORT_DESC
                up = !up;
#endif

                const bool after = afterAsc(sharedValue[idx], sharedIndex[idx], sharedValue[ixj], sharedIndex[ixj]);
                if (up == after) {
                    const DTYPE tValue = sharedValue[idx];
                    sharedValue[idx] = sharedValue[ixj];
                    sharedValue[ixj] = tValue;
                    const int tIndex = sharedIndex[idx];
                    sharedIndex[idx] = sharedIndex[ixj];
                    sharedIndex[ixj] = tIndex;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (tid == 0) {
        __global DTYPE *rowOut = outValue + row * k;
        __global int *rowIdx = outIndex + row * k;
        const int realK = min(k, rowSize);
        for (int i = 0; i < realK; ++i) {
            rowOut[i] = sharedValue[i];
            rowIdx[i] = sharedIndex[i];
        }
    }
}
