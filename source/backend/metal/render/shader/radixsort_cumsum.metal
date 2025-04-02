#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;


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

kernel void main0(device pointoffsetSum& uPointoffsetSum [[buffer(0)]], const device pointoffset& uPointoffset [[buffer(1)]], constant constBuffer& uConstant [[buffer(2)]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])
{
    threadgroup uint local_sum[LOCAL_SIZE];
    int tId = int(gl_LocalInvocationID.x);
    int size = (uConstant.point.x + 3) / 4;
    int curOffset = 0;
    uint sum = 0u;
    uint4 threadBuffer[UNIT];
    uint _233;
    while (curOffset < size)
    {
        int sta = (tId * UNIT) + curOffset;
        int fin = min((sta + UNIT), size);
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
            for (uint stride = 1u; stride <= LOCAL_SIZE / 2u; stride *= 2u)
            {
                uint id = ((uint(tId + 1) * stride) * 2u) - 1u;
                if (id < LOCAL_SIZE)
                {
                    local_sum[id] += local_sum[id - stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint stride_1 = LOCAL_SIZE / 4u; stride_1 > 0u; stride_1 /= 2u)
            {
                uint id_1 = ((uint(tId + 1) * stride_1) * 2u) - 1u;
                if ((id_1 + stride_1) < LOCAL_SIZE)
                {
                    uint _220 = id_1 + stride_1;
                    local_sum[_220] += local_sum[id_1];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tId > 0)
            {
                _233 = local_sum[tId - 1];
            }
            else
            {
                _233 = 0u;
            }
            uint sum0 = _233;
            for (int i_2 = sta; i_2 < fin; i_2++)
            {
                int lpos_2 = i_2 - sta;
                uPointoffsetSum.data[i_2] = threadBuffer[lpos_2] + uint4(sum + sum0);
            }
            sum += local_sum[LOCAL_SIZE - 1];
        }
        curOffset += LOCAL_SIZE * UNIT;
        if (curOffset < size)
        {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

