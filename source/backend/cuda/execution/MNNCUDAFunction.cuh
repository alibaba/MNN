#ifndef MNNCUDAFunction_cuh
#define MNNCUDAFunction_cuh

#include <stdint.h>

struct DivModFast {
    DivModFast(int d = 1)
    {
        d_ = (d == 0) ? 1 : d;
        for (l_ = 0;; ++l_) {
            if ((1U << l_) >= d_)
                break;
        }
        uint64_t one = 1;
        uint64_t m   = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
        m_           = static_cast<uint32_t>(m);
    }

    __device__ __inline__ int div(int idx) const
    {
        uint32_t tm = __umulhi(m_, idx); // get high 32-bit of the product
        return (tm + idx) >> l_;
    }

    __device__ __inline__ int mod(int idx) const
    {
        return idx - d_ * div(idx);
    }

    __device__ __inline__ void divmod(int idx, int &quo, int &rem)
    {
        quo = div(idx);
        rem = idx - quo * d_;
    }

    uint32_t d_; // divisor
    uint32_t l_; // ceil(log2(d_))
    uint32_t m_; // m' in the papaer
};


#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
    for(int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    }
    return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if(lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
    val = warpReduceSum(val);
    return val;
}

template <typename T>
__inline__ __device__
T warpReduceMax(T val)
{
    for(int mask = 16; mask > 0; mask >>= 1) {
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    }
    return val;
}

template <typename T>
__inline__ __device__
T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax<T>(val);

    if(lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
    val = warpReduceMax(val);
    return val;
}

#endif