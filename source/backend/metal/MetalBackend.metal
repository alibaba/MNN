//
//  MetalBackend.metal
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct tensor_shape {
    int size;
    int channel;
    int slice;
    int batch_slices;
};

kernel void version_func_001(const device uchar *in [[buffer(0)]],
                                  device uchar *out     [[buffer(1)]],
                                  uint gid              [[thread_position_in_grid]]) {
    // do nothing, just for verifying match between mnn and metallib
}

kernel void copy_byte(const device uchar *in    [[buffer(0)]],
                      device uchar *out         [[buffer(1)]],
                      uint gid                  [[thread_position_in_grid]]) {
    out[int(gid)] = in[int(gid)];
}
kernel void copy_int(const device int *in       [[buffer(0)]],
                     device int *out            [[buffer(1)]],
                     uint gid                   [[thread_position_in_grid]]) {
    out[int(gid)] = in[int(gid)];
}
kernel void copy_float(const device ftype *in   [[buffer(0)]],
                       device ftype *out        [[buffer(1)]],
                       uint gid                 [[thread_position_in_grid]]) {
    out[int(gid)] = in[int(gid)];
}

kernel void upcast_float(const device ftype *in     [[buffer(0)]],
                         device float *out          [[buffer(1)]],
                         uint gid                   [[thread_position_in_grid]]) {
    out[int(gid)] = in[int(gid)];
}
kernel void downcast_float(const device float *in   [[buffer(0)]],
                           device ftype *out        [[buffer(1)]],
                           uint gid                 [[thread_position_in_grid]]) {
    out[int(gid)] = in[int(gid)];
}
struct Limit {
    uint4 size;
};
kernel void upcast_float4(const device ftype4 *in   [[buffer(0)]],
                          device float4 *out        [[buffer(1)]],
                          constant Limit& limit [[buffer(2)]],
                          uint gid                  [[thread_position_in_grid]]) {
    if (gid < limit.size.x) {
        out[int(gid)] = float4(in[int(gid)]);
    }
}
kernel void downcast_float4(const device float4 *in [[buffer(0)]],
                            device ftype4 *out      [[buffer(1)]],
                            constant Limit& limit [[buffer(2)]],
                            uint gid                [[thread_position_in_grid]]) {
    if (gid < limit.size.x) {
        out[int(gid)] = ftype4(in[int(gid)]);
    }
}

template <typename IType, typename OType>
static inline void template_NHWC_to_NC4HW4(const device IType *in, device OType *out, constant tensor_shape &s, uint2 gid) {
    int b = gid.y / s.slice;
    int z = gid.y % s.slice;
    int c = z * 4;
    
    auto off_in  = in  + b          * s.size * s.channel + int(gid.x) * s.channel + c;
    auto off_out = out + int(gid.y) * s.size             + int(gid.x);
    off_out[0] = OType(c + 0 < s.channel ? off_in[0] : 0,
                       c + 1 < s.channel ? off_in[1] : 0,
                       c + 2 < s.channel ? off_in[2] : 0,
                       c + 3 < s.channel ? off_in[3] : 0);
}
kernel void upcast_f_NHWC_to_NC4HW4(const device ftype *in      [[buffer(0)]],
                                    device float4 *out          [[buffer(1)]],
                                    constant tensor_shape &s    [[buffer(2)]],
                                    uint2 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NHWC_to_NC4HW4<ftype, float4>(in, out, s, gid);
}
kernel void downcast_f_NHWC_to_NC4HW4(const device float *in    [[buffer(0)]],
                                      device ftype4 *out        [[buffer(1)]],
                                      constant tensor_shape &s  [[buffer(2)]],
                                      uint2 gid                 [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NHWC_to_NC4HW4<float, ftype4>(in, out, s, gid);
}
kernel void cvt_u_NHWC_to_NC4HW4(const device uchar *in     [[buffer(0)]],
                                 device uchar4 *out         [[buffer(1)]],
                                 constant tensor_shape &s   [[buffer(2)]],
                                 uint2 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NHWC_to_NC4HW4<uchar, uchar4>(in, out, s, gid);
}
kernel void cvt_f_NHWC_to_NC4HW4(const device ftype *in     [[buffer(0)]],
                                 device ftype4 *out         [[buffer(1)]],
                                 constant tensor_shape &s   [[buffer(2)]],
                                 uint2 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NHWC_to_NC4HW4<ftype, ftype4>(in, out, s, gid);
}

template <typename IType, typename OType>
static inline void template_NC4HW4_to_NHWC(const device IType *in, device OType *out, constant tensor_shape &s, uint2 gid) {
    int b = gid.y / s.slice;
    int z = gid.y % s.slice;
    int c = z * 4;
    auto off_in  = in  + int(gid.y) * s.size             + int(gid.x);
    auto off_out = out + b          * s.size * s.channel + int(gid.x) * s.channel + c;
    
    IType v4 = off_in[0];
    /* if (1)           */ off_out[0] = v4[0];
    if (c + 1 < s.channel) off_out[1] = v4[1];
    if (c + 2 < s.channel) off_out[2] = v4[2];
    if (c + 3 < s.channel) off_out[3] = v4[3];
}
kernel void upcast_f_NC4HW4_to_NHWC(const device ftype4 *in     [[buffer(0)]],
                                    device float *out           [[buffer(1)]],
                                    constant tensor_shape &s    [[buffer(2)]],
                                    uint2 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NC4HW4_to_NHWC<ftype4, float>(in, out, s, gid);
}
kernel void downcast_f_NC4HW4_to_NHWC(const device float4 *in   [[buffer(0)]],
                                      device ftype *out         [[buffer(1)]],
                                      constant tensor_shape &s  [[buffer(2)]],
                                      uint2 gid                 [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NC4HW4_to_NHWC<float4, ftype>(in, out, s, gid);
}
kernel void cvt_u_NC4HW4_to_NHWC(const device uchar4 *in    [[buffer(0)]],
                                 device uchar *out          [[buffer(1)]],
                                 constant tensor_shape &s   [[buffer(2)]],
                                 uint2 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NC4HW4_to_NHWC<uchar4, uchar>(in, out, s, gid);
}
kernel void cvt_f_NC4HW4_to_NHWC(const device ftype4 *in    [[buffer(0)]],
                                 device ftype *out          [[buffer(1)]],
                                 constant tensor_shape &s   [[buffer(2)]],
                                 uint2 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NC4HW4_to_NHWC<ftype4, ftype>(in, out, s, gid);
}

template <typename IType, typename OType>
static inline void template_NCHW_to_NC4HW4(const device IType *in, device OType *out, constant tensor_shape &s, uint2 gid) {
    int b = gid.y / s.slice;
    int z = gid.y % s.slice;
    int c = z * 4;
    
    auto off_in  = in  + (b * s.channel + c) * s.size + int(gid.x);
    auto off_out = out + int(gid.y)          * s.size + int(gid.x);
    off_out[0] = OType(c + 0 < s.channel ? off_in[0 * s.size] : 0.0h,
                       c + 1 < s.channel ? off_in[1 * s.size] : 0.0h,
                       c + 2 < s.channel ? off_in[2 * s.size] : 0.0h,
                       c + 3 < s.channel ? off_in[3 * s.size] : 0.0h);
}
kernel void upcast_f_NCHW_to_NC4HW4(const device ftype *in      [[buffer(0)]],
                                    device float4 *out          [[buffer(1)]],
                                    constant tensor_shape &s    [[buffer(2)]],
                                    uint2 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NCHW_to_NC4HW4<ftype, float4>(in, out, s, gid);
}
kernel void downcast_f_NCHW_to_NC4HW4(const device float *in    [[buffer(0)]],
                                      device ftype4 *out        [[buffer(1)]],
                                      constant tensor_shape &s  [[buffer(2)]],
                                      uint2 gid                 [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NCHW_to_NC4HW4<float, ftype4>(in, out, s, gid);
}
kernel void cvt_u_NCHW_to_NC4HW4(const device uchar *in     [[buffer(0)]],
                                 device uchar4 *out         [[buffer(1)]],
                                 constant tensor_shape &s   [[buffer(2)]],
                                 uint2 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NCHW_to_NC4HW4<uchar, uchar4>(in, out, s, gid);
}
kernel void cvt_f_NCHW_to_NC4HW4(const device ftype *in     [[buffer(0)]],
                                 device ftype4 *out         [[buffer(1)]],
                                 constant tensor_shape &s   [[buffer(2)]],
                                 uint2 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NCHW_to_NC4HW4<ftype, ftype4>(in, out, s, gid);
}

template <typename IType, typename OType>
static inline void template_NC4HW4_to_NCHW(const device IType *in, device OType *out, constant tensor_shape &s, uint2 gid) {
    int b = gid.y / s.slice;
    int z = gid.y % s.slice;
    int c = z * 4;
    
    auto off_in  = in  + int(gid.y)          * s.size + int(gid.x);
    auto off_out = out + (b * s.channel + c) * s.size + int(gid.x);
    IType v4 = off_in[0];
    /* if (1)           */ off_out[0 * s.size] = v4.x;
    if (c + 1 < s.channel) off_out[1 * s.size] = v4.y;
    if (c + 2 < s.channel) off_out[2 * s.size] = v4.z;
    if (c + 3 < s.channel) off_out[3 * s.size] = v4.w;
}
kernel void upcast_f_NC4HW4_to_NCHW(const device ftype4 *in     [[buffer(0)]],
                                    device float *out           [[buffer(1)]],
                                    constant tensor_shape &s    [[buffer(2)]],
                                    uint2 gid                   [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NC4HW4_to_NCHW<ftype4, float>(in, out, s, gid);
}
kernel void downcast_f_NC4HW4_to_NCHW(const device float4 *in   [[buffer(0)]],
                                      device ftype *out         [[buffer(1)]],
                                      constant tensor_shape &s  [[buffer(2)]],
                                      uint2 gid                 [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NC4HW4_to_NCHW<float4, ftype>(in, out, s, gid);
}
kernel void cvt_u_NC4HW4_to_NCHW(const device uchar4 *in    [[buffer(0)]],
                                 device uchar *out          [[buffer(1)]],
                                 constant tensor_shape &s   [[buffer(2)]],
                                 uint2 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NC4HW4_to_NCHW<uchar4, uchar>(in, out, s, gid);
}
kernel void cvt_f_NC4HW4_to_NCHW(const device ftype4 *in    [[buffer(0)]],
                                 device ftype *out          [[buffer(1)]],
                                 constant tensor_shape &s   [[buffer(2)]],
                                 uint2 gid                  [[thread_position_in_grid]]) {
    if ((int)gid.x < s.size && (int)gid.y < s.batch_slices) template_NC4HW4_to_NCHW<ftype4, ftype>(in, out, s, gid);
}
struct SamplerInfo {
    uint4 stride;//stride[3] + offset
    uint4 size;//size[3] + totalSize
    uint4 extent;//dstStride[3]+dstOffset
    uint4 imageSize;
};

kernel void blit_intx4(const device int4 *in   [[buffer(0)]],
                       device int4 *out        [[buffer(1)]],
                       constant SamplerInfo &info        [[buffer(2)]],
                       uint3 gid                 [[thread_position_in_grid]]) {
    if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
        uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
        uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
        out[int(dstOffset)] = in[int(srcOffset)];
    }
}
kernel void blit_int(const device int *in   [[buffer(0)]],
                       device int *out        [[buffer(1)]],
                       constant SamplerInfo &info        [[buffer(2)]],
                       uint3 gid                 [[thread_position_in_grid]]) {
    if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
        uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
        uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
        out[int(dstOffset)] = in[int(srcOffset)];
    }
}
kernel void blit_int8(const device char *in   [[buffer(0)]],
                       device char *out        [[buffer(1)]],
                       constant SamplerInfo &info        [[buffer(2)]],
                       uint3 gid                 [[thread_position_in_grid]]) {
    if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
        uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
        uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
        out[int(dstOffset)] = in[int(srcOffset)];
    }
}

kernel void blit_int16(const device short *in   [[buffer(0)]],
                       device short *out        [[buffer(1)]],
                       constant SamplerInfo &info        [[buffer(2)]],
                       uint3 gid                 [[thread_position_in_grid]]) {
    if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
        uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
        uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
        out[int(dstOffset)] = in[int(srcOffset)];
    }
}

kernel void blit_int64(const device short4 *in   [[buffer(0)]],
                       device short4 *out        [[buffer(1)]],
                       constant SamplerInfo &info        [[buffer(2)]],
                       uint3 gid                 [[thread_position_in_grid]]) {
    if (gid.x < info.size.x && gid.y < info.size.y && gid.z < info.size.z) {
        uint dstOffset = gid.x * info.extent.x + gid.y * info.extent.y + gid.z * info.extent.z + info.extent.w;
        uint srcOffset = gid.x * info.stride.x + gid.y * info.stride.y + gid.z * info.stride.z + info.stride.w;
        out[int(dstOffset)] = in[int(srcOffset)];
    }
}
