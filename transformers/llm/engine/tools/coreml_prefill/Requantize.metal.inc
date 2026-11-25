R"metal(
#include <metal_stdlib>
using namespace metal;
struct Params {
    uint rows, cols, raw, alpha, width, pitch;
};
kernel void requantLinear(device const uchar* src [[buffer(0)]], device half* scales [[buffer(1)]],
                          constant Params& p [[buffer(2)]], device uchar4* out [[buffer(3)]],
                          uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                          uint tid [[thread_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]],
                          uint3 threads [[threads_per_threadgroup]]) {
    const uint row = group.x, step = threads.x;
    const device half* alpha = (const device half*)(src + p.alpha) + (row * p.cols / 32);
    const device uchar* codes = src + p.raw + row * (p.cols / 2);
    threadgroup float partial[32];
    threadgroup float rowscale;
    float peak = 0.0f;
    for (uint j = tid; j < p.cols / 4; j += step) {
        uchar2 v = uchar2(codes[2 * j], codes[2 * j + 1]);
        uchar4 n = uchar4(v >> 4, v & 15);
        float2 bounds = float2(min(min(n.x, n.y), min(n.z, n.w)), max(max(n.x, n.y), max(n.z, n.w))) - 8.0f;
        float2 a = float2(alpha[(j / 16) * 2], alpha[(j / 16) * 2 + 1]);
        float2 val = abs(fma(bounds, a.y, a.x));
        peak = max(peak, max(val.x, val.y));
    }
    float local = simd_max(peak);
    if (!lane) {
        partial[simd] = local;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!simd) {
        local = simd_max(lane < step / 32 ? partial[lane] : 0.0f);
        if (!lane) {
            half h = half(local == 0 ? 1.0f : max(local / 127.0f, 0x1p-24f));
            rowscale = float(h);
            scales[row] = h;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < p.cols / 4; j += step) {
        uchar2 v = uchar2(codes[2 * j], codes[2 * j + 1]);
        uint a = (j / 16) * 2;
        float val = fma(float(lane % 16) - 8.0f, float(alpha[a + 1]), float(alpha[a]));
        uint q = uint(clamp(rint(val / rowscale), -127.0f, 127.0f) + 128.0f);
        ushort base = ushort(lane & 16u);
        uchar4 result = uchar4(simd_shuffle(q, base + ushort(v.x >> 4)), simd_shuffle(q, base + ushort(v.x & 15)),
                               simd_shuffle(q, base + ushort(v.y >> 4)), simd_shuffle(q, base + ushort(v.y & 15)));
        uint offset = row * p.cols + 4 * j;
        out[((offset / p.width) * p.pitch + offset % p.width) / 4] = result;
    }
}
)metal"
