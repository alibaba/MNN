#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct constBuffer
{
    int4 size;
    float4 scale;
};

struct destBuffer
{
    float data[1];
};

struct sourceBuffer0
{
    int data[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(256u, 1u, 1u);

kernel void main0(device destBuffer& uOutput [[buffer(0)]], const device sourceBuffer0& uInput [[buffer(1)]], constant constBuffer& uConstant [[buffer(2)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int pos = int(gl_GlobalInvocationID.x);
    int4 size = uConstant.size;
    if (pos < size.x)
    {
        uOutput.data[pos] = (float(uInput.data[pos]) * uConstant.scale.x) + uConstant.scale.y;
    }
}

