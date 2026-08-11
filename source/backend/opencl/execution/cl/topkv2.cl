#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define THREAD_NUMBER 128
#define LOCAL_K 8
#define CANDIDATE_NUMBER (THREAD_NUMBER * LOCAL_K)

#ifdef IS_INT
typedef int DTYPE;
typedef int4 DTYPE4;
#define READ_INPUT(img, coord) read_imagei(img, SAMPLER, coord)
#define WRITE_OUTPUT(img, coord, val) write_imagei(img, coord, val)
#else
typedef FLOAT DTYPE;
typedef FLOAT4 DTYPE4;
#define READ_INPUT(img, coord) RI_F(img, SAMPLER, coord)
#define WRITE_OUTPUT(img, coord, val) WI_F(img, coord, val)
#endif

inline bool afterAsc(DTYPE aValue, int aIndex, DTYPE bValue, int bIndex) {
    if (aValue > bValue) return true;
    if (aValue < bValue) return false;
    return aIndex > bIndex;
}

inline bool better(DTYPE aValue, int aIndex, DTYPE bValue, int bIndex) {
    if (bIndex < 0) return true;
    if (aIndex < 0) return false;
#ifdef SORT_DESC
    if (aValue > bValue) return true;
    if (aValue < bValue) return false;
#else
    if (aValue < bValue) return true;
    if (aValue > bValue) return false;
#endif
    return aIndex < bIndex;
}

// Sort along channel dimension (last dim for 1D/2D tensors)
// Image layout: image_x = w + c4_idx * width, image_y = n * height + h
// For 2D [M, N]: width=1, height=1, numRows=M, rowSize=N
// For 1D [N]: width=1, height=1, numRows=1, rowSize=N
__kernel void topkv2_channel(GLOBAL_SIZE_3_DIMS
    __read_only image2d_t input,
    __write_only image2d_t outputValue,
    __write_only image2d_t outputIndex,
    __private const int rowSize,
    __private const int k,
    __private const int width,
    __private const int channelBlocks
) {
    const int gid0 = get_global_id(0);
    const int gid1 = get_global_id(1);
    const int gid2 = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(gid0, gid1, gid2);

    const int tid = get_local_id(0);
    const int row = get_group_id(1);

    // Decompose row into image coordinates
    // row = image_y * width + w_pos (for sort-along-C)
    const int w_pos = row % width;
    const int image_y = row / width;

    // Initialize per-thread top-k
#ifdef IS_INT
#ifdef SORT_DESC
    DTYPE initVal = (DTYPE)(-2147483647 - 1);
#else
    DTYPE initVal = (DTYPE)(2147483647);
#endif
#else
#ifdef SORT_DESC
    DTYPE initVal = (DTYPE)(-FLT_MAX);
#else
    DTYPE initVal = (DTYPE)(FLT_MAX);
#endif
#endif
    DTYPE localValue[LOCAL_K];
    int localIndex[LOCAL_K];
    for (int i = 0; i < LOCAL_K; ++i) {
        localValue[i] = initVal;
        localIndex[i] = -1;
    }

    // Read input image and build per-thread top-k
    for (int c4 = tid; c4 < channelBlocks; c4 += THREAD_NUMBER) {
        DTYPE4 pixel = READ_INPUT(input, (int2)(c4 * width + w_pos, image_y));
        int baseC = c4 << 2;

        DTYPE vals[4];
        vals[0] = pixel.x;
        vals[1] = pixel.y;
        vals[2] = pixel.z;
        vals[3] = pixel.w;

        for (int comp = 0; comp < 4 && baseC + comp < rowSize; ++comp) {
            DTYPE value = vals[comp];
            int idx = baseC + comp;

            if (!better(value, idx, localValue[LOCAL_K - 1], localIndex[LOCAL_K - 1])) continue;

            int insertPos = LOCAL_K;
            for (int j = 0; j < LOCAL_K; ++j) {
                if (better(value, idx, localValue[j], localIndex[j])) {
                    insertPos = j;
                    break;
                }
            }
            if (insertPos >= LOCAL_K) continue;

            for (int j = LOCAL_K - 1; j > insertPos; --j) {
                localValue[j] = localValue[j - 1];
                localIndex[j] = localIndex[j - 1];
            }
            localValue[insertPos] = value;
            localIndex[insertPos] = idx;
        }
    }

    // Scatter to shared memory
    __local DTYPE sharedValue[CANDIDATE_NUMBER];
    __local int sharedIndex[CANDIDATE_NUMBER];
    const int base = tid * LOCAL_K;
    for (int i = 0; i < LOCAL_K; ++i) {
        sharedValue[base + i] = localValue[i];
        sharedIndex[base + i] = localIndex[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Bitonic sort in shared memory
    for (uint size = 2; size <= CANDIDATE_NUMBER; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint sIdx = tid; sIdx < CANDIDATE_NUMBER; sIdx += THREAD_NUMBER) {
                const uint ixj = sIdx ^ stride;
                if (ixj <= sIdx) continue;

                bool up = ((sIdx & size) == 0);
#ifdef SORT_DESC
                up = !up;
#endif
                const bool after = afterAsc(sharedValue[sIdx], sharedIndex[sIdx],
                                            sharedValue[ixj], sharedIndex[ixj]);
                if (up == after) {
                    DTYPE tValue = sharedValue[sIdx];
                    sharedValue[sIdx] = sharedValue[ixj];
                    sharedValue[ixj] = tValue;
                    int tIndex = sharedIndex[sIdx];
                    sharedIndex[sIdx] = sharedIndex[ixj];
                    sharedIndex[ixj] = tIndex;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Thread 0 writes output
    if (tid == 0) {
        const int realK = min(k, rowSize);
        const int outChannelBlocks = (k + 3) >> 2;

        for (int c4 = 0; c4 < outChannelBlocks; ++c4) {
            DTYPE4 valuePixel = (DTYPE4)(0);
            int4 indexPixel = (int4)(0);
            int baseIdx = c4 << 2;

            if (baseIdx < realK) {
                valuePixel.x = sharedValue[baseIdx];
                indexPixel.x = sharedIndex[baseIdx];
            }
            if (baseIdx + 1 < realK) {
                valuePixel.y = sharedValue[baseIdx + 1];
                indexPixel.y = sharedIndex[baseIdx + 1];
            }
            if (baseIdx + 2 < realK) {
                valuePixel.z = sharedValue[baseIdx + 2];
                indexPixel.z = sharedIndex[baseIdx + 2];
            }
            if (baseIdx + 3 < realK) {
                valuePixel.w = sharedValue[baseIdx + 3];
                indexPixel.w = sharedIndex[baseIdx + 3];
            }

            int outX = c4 * width + w_pos;
            WRITE_OUTPUT(outputValue, (int2)(outX, image_y), valuePixel);
            write_imagei(outputIndex, (int2)(outX, image_y), indexPixel);
        }
    }
}
