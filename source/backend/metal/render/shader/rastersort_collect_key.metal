#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct constBuffer
{
    uint4 point;
};

struct histogram
{
    uint data[1];
};

struct pointkeys
{
    uint2 data[1];
};

struct sourceBuffer1
{
    float4 data[1];
};

struct sourceBuffer0
{
    TYPE data[1];
};

#ifndef SPIRV_CROSS_CONSTANT_ID_0
#define SPIRV_CROSS_CONSTANT_ID_0 1u
#endif
constant uint _210 = SPIRV_CROSS_CONSTANT_ID_0;
#ifndef SPIRV_CROSS_CONSTANT_ID_1
#define SPIRV_CROSS_CONSTANT_ID_1 1u
#endif
constant uint _211 = SPIRV_CROSS_CONSTANT_ID_1;
#ifndef SPIRV_CROSS_CONSTANT_ID_2
#define SPIRV_CROSS_CONSTANT_ID_2 1u
#endif
constant uint _212 = SPIRV_CROSS_CONSTANT_ID_2;
constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(_210, _211, _212);

kernel void main0(device pointkeys& uPointKeys [[buffer(0)]], const device sourceBuffer0& uAttr [[buffer(1)]], const device sourceBuffer1& uViewProj [[buffer(2)]], const device histogram& uHistogram [[buffer(3)]], constant constBuffer& uConstant [[buffer(4)]], uint3 gl_NumWorkGroups [[threadgroups_per_grid]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    uint groupSize = gl_NumWorkGroups.x;
    uint threadId = gl_GlobalInvocationID.x;
    uint threadNumber = groupSize * LOCAL_SIZE;
    uint totalSize = uConstant.point.x;
    uint size = ((totalSize + threadNumber) - 1u) / threadNumber;
    uint sta = threadId * size;
    uint fin = min((sta + size), totalSize);
    uint offset = 0u;
    if (threadId > 0u)
    {
        offset = uHistogram.data[threadId - 1u];
    }
    uint sortSize = uHistogram.data[threadNumber - 1u];
    if (threadId == 0u)
    {
        if ((sortSize % 2u) > 0u)
        {
            ((device uint*)&uPointKeys.data[sortSize])[0u] = 32767u;
            ((device uint*)&uPointKeys.data[sortSize])[1u] = 0u;
        }
    }
    uint2 kv;
    for (uint pos = sta; pos < fin; pos++)
    {
        float4 vp0 = uViewProj.data[0];
        float4 vp1 = uViewProj.data[1];
        float4 vp2 = uViewProj.data[2];
        float4 vp3 = uViewProj.data[3];
        float4 attr = float4(uAttr.data[pos]);
        float depth = (((attr.x * vp0.z) + (attr.y * vp1.z)) + (attr.z * vp2.z)) + vp3.z;
        float dw = (((attr.x * vp0.w) + (attr.y * vp1.w)) + (attr.z * vp2.w)) + vp3.w;
        depth /= dw;
        if ((depth >= 0.0) && (depth <= 1.0))
        {
            kv.x = uint(depth * 32767.0);
            kv.y = pos;
            uPointKeys.data[offset] = kv;
            offset++;
        }
    }
}

