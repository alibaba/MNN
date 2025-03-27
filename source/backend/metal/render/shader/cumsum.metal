#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct constBuffer
{
    int4 point;
};

struct pointoffset
{
    uint4 data[1];
};

struct pointoffsetSum
{
    uint4 data[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(256u, 1u, 1u);

kernel void main0(device pointoffsetSum& uPointoffsetSum [[buffer(0)]], const device pointoffset& uPointoffset [[buffer(1)]], constant constBuffer& uConstant [[buffer(2)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])
{
    threadgroup uint local_sum[256];
    int tId = int(gl_LocalInvocationID.x);
    int size = (uConstant.point.x + 3) / 4;
    int curOffset = 0;
    uint sum = 0u;
    spvUnsafeArray<uint4, 128> threadBuffer;
    while (curOffset < size)
    {
        int sta = (tId * 128) + curOffset;
        int fin = min((sta + 128), size);
        for (int i = sta; i < fin; i++)
        {
            int lpos = i - sta;
            uint4 p0 = uPointoffset.data[i];
            p0.y += p0.x;
            p0.z += p0.y;
            p0.w += p0.z;
            threadBuffer[lpos] = p0;
        }
        int _112 = sta + 1;
        for (int i_1 = _112; i_1 < fin; i_1++)
        {
            int lpos_1 = i_1 - sta;
            uint4 p0_1 = threadBuffer[lpos_1];
            uint4 p1 = threadBuffer[lpos_1 - 1];
            p0_1 += uint4(p1.w);
            threadBuffer[lpos_1] = p0_1;
        }
        local_sum[tId] = threadBuffer[(fin - sta) - 1].w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (fin > sta)
        {
            for (int i_2 = 0; i_2 < tId; i_2++)
            {
                sum += local_sum[i_2];
            }
            for (int i_3 = sta; i_3 < fin; i_3++)
            {
                int lpos_2 = i_3 - sta;
                uPointoffsetSum.data[i_3] = threadBuffer[lpos_2] + uint4(sum);
            }
            for (int i_4 = tId; i_4 < 256; i_4++)
            {
                sum += local_sum[i_4];
            }
        }
        curOffset += 32768;
        if (curOffset < size)
        {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

