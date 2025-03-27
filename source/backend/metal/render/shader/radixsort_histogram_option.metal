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

struct variableBuffer
{
    uint4 off;
};

struct variablepBuffer
{
    uint4 off;
};

struct pointO
{
    uint4 data[1];
};

struct pointI
{
    uint data[1];
};


kernel void main0(device pointI& uHistogram [[buffer(0)]], const device pointO& uPointKeysInput [[buffer(1)]], constant variableBuffer& uOffset [[buffer(2)]], constant variablepBuffer& uPass [[buffer(3)]], uint3 gl_NumWorkGroups [[threadgroups_per_grid]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    uint groupSize = gl_NumWorkGroups.x;
    uint threadId = gl_GlobalInvocationID.x;
    spvUnsafeArray<uint, BIN_NUMBER> binSize;
    for (int i = 0; i < BIN_NUMBER; i++)
    {
        binSize[i] = 0u;
    }
    uint totalSize = (uOffset.off.x + 1u) / 2u;
    uint threadNumber = groupSize * LOCAL_SIZE;
    uint size = ((totalSize + threadNumber) - 1u) / threadNumber;
    uint sta = threadId * size;
    uint fin = min((sta + size), totalSize);
    uint div = uPass.off.x;
    uint modNum = BIN_NUMBER - 1u;
    for (uint i_1 = sta; i_1 < fin; i_1++)
    {
        uint2 key = (uPointKeysInput.data[i_1].xz >> uint2(div)) & uint2(modNum);
        binSize[key.x]++;
        binSize[key.y]++;
    }
    for (int i_2 = 0; i_2 < BIN_NUMBER; i_2++)
    {
        uHistogram.data[(uint(i_2) * threadNumber) + threadId] = binSize[i_2];
    }
}

