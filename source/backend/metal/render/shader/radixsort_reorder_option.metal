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

struct his
{
    uint data[1];
};

struct pointO
{
    uint2 data[1];
};

struct pointI
{
    uint2 data[1];
};


kernel void main0(device pointI& uPointKeysOutput [[buffer(0)]], const device pointO& uPointKeysInput [[buffer(1)]], const device his& uHistogram [[buffer(2)]], constant variableBuffer& uOffset [[buffer(3)]], constant variablepBuffer& uPass [[buffer(4)]], uint3 gl_NumWorkGroups [[threadgroups_per_grid]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    uint groupSize = gl_NumWorkGroups.x;
    uint threadId = gl_GlobalInvocationID.x;
    uint threadNumber = groupSize * LOCAL_SIZE;
    uint totalSize = (uOffset.off.x + 1u) / 2u;
    uint size = ((totalSize + threadNumber) - 1u) / threadNumber;
    uint sta = threadId * size;
    uint fin = min((sta + size), totalSize);
    uint div = uPass.off.x;
    sta *= 2u;
    fin *= 2u;
    uint modNum = BIN_NUMBER - 1u;
    spvUnsafeArray<uint, BIN_NUMBER> offsets;
    for (int i = 0; i < BIN_NUMBER; i++)
    {
        uint pos = (uint(i) * threadNumber) + threadId;
        if (pos == 0u)
        {
            offsets[i] = 0u;
        }
        else
        {
            offsets[i] = uHistogram.data[pos - 1u];
        }
    }
    for (uint i_1 = sta; i_1 < fin; i_1++)
    {
        uint2 value = uPointKeysInput.data[i_1];
        uint key = (value.x >> div) & modNum;
        uint pos_1 = offsets[key];
        uPointKeysOutput.data[pos_1] = value;
        offsets[key]++;
    }
}

