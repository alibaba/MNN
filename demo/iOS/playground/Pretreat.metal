#include <metal_stdlib>

using namespace metal;

struct PretreatInfo {
    int4 outputSize;
    float4 mean;
    float4 normal;
    float4 inputSize;
    float4x4 matrix;
};

kernel void pretreat(device half4* output  [[buffer(0)]],
                     texture2d<float> input [[texture(0)]],
                     constant PretreatInfo& info[[buffer(1)]],
                   uint2 gid[[thread_position_in_grid]]) {
    constexpr sampler linearSampler(mip_filter::none,
                                    mag_filter::linear,
                                    min_filter::linear);
    if ((int)gid.x < (int)info.outputSize.x && (int)gid.y < (int)info.outputSize.y) {
#ifdef COMMON_MATRIX
        float3 pos = float3((float)gid.x, (float)gid.y, 1.0);
        pos = pos * info.matrix;
        float4 color = input.sample(linearSampler, pos.xy / pos.z);
#else
        float2 pos = float2((float)(gid.x - 1) / (float)(info.outputSize.x - 1), (float)(gid.y - 1) / (float)(info.outputSize.y - 1));
        float4 color = input.sample(linearSampler, pos);
#endif
        color = (color * float4(255) - info.mean) * info.normal;
        output[gid.y * info.outputSize.x + gid.x] = half4(color);
    }
}
