//
//  MetalCropAndResize.metal
//  MNN
//
//  Created by MNN on 2018/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalDefine.metal"

using namespace metal;

struct crop_and_resize_shape {
    int image_count; // batch
    int image_height;
    int image_width;
    int image_channel;
    int boxes_count; // batch
    int crop_height;
    int crop_width;
    int crop_channel;
    float extrapolation;
};

kernel void crop_and_resize_bilinear(const device ftype *image          [[buffer(0)]],
                                     const device ftype4 *boxes         [[buffer(1)]],
                                     const device int *indexes          [[buffer(2)]],
                                     device ftype *crops                [[buffer(3)]],
                                     constant crop_and_resize_shape &s  [[buffer(4)]],
                                     uint3 gid                          [[thread_position_in_grid]]) {
    // grid protect
    int cb = gid.z / s.crop_height;
    int ch = gid.z % s.crop_height, cw = gid.y, cc = gid.x;
    if (cb >= s.boxes_count || ch >= s.crop_height || cw >= s.crop_width || cc >= s.crop_channel) return;
    
    // image index protect
    auto image_index = indexes[cb];
    if (0 > image_index || image_index >= s.image_count) return;
    
    // prepare const
    auto rect = boxes[cb];
    auto y1 = rect[0], x1 = rect[1], y2 = rect[2], x2 = rect[3];
    auto crop = crops
        + cb * s.crop_height * s.crop_width * s.crop_channel
        + ch                 * s.crop_width * s.crop_channel
        + cw                                * s.crop_channel
        + cc                                                ;
    
    // image y protect
    auto height_scale = s.crop_height > 1 ? (y2 - y1) * (s.image_height - 1) / (s.crop_height - 1) : 0;
    float im_y = (s.crop_height > 1) ? y1 * (s.image_height - 1) + ch * height_scale : 0.5f * (y1 + y2) * (s.image_height - 1);
    if (0 > im_y || im_y > s.image_height - 1) {
        *crop = ftype(s.extrapolation);
        return;
    }
    
    // image x protect
    auto width_scale = s.crop_width > 1 ? (x2 - x1) * (s.image_width - 1) / (s.crop_width - 1) : 0;
    float im_x = (s.crop_width > 1) ? x1 * (s.image_width - 1) + cw * width_scale : 0.5f * (x1 + x2) * (s.image_width - 1);
    if (0 > im_x || im_x > s.image_width - 1) {
        *crop = ftype(s.extrapolation);
        return;
    }
    
    // get bilinear
    auto cc_image = image + image_index * s.image_height * s.image_width * s.image_channel + cc;
    int top_y = floor(im_y), btm_y = ceil(im_y);
    int lft_x = floor(im_x), rgt_x = ceil(im_x);
    auto tl = cc_image[top_y * s.image_width * s.image_channel + lft_x * s.image_channel];
    auto tr = cc_image[top_y * s.image_width * s.image_channel + rgt_x * s.image_channel];
    auto bl = cc_image[btm_y * s.image_width * s.image_channel + lft_x * s.image_channel];
    auto br = cc_image[btm_y * s.image_width * s.image_channel + rgt_x * s.image_channel];
    
    auto y_lerp = im_y - top_y, x_lerp = im_x - lft_x;
    auto top = tl + (tr - tl) * x_lerp;
    auto btm = bl + (br - bl) * x_lerp;
    *crop = top + (btm - top) * y_lerp;
}

kernel void crop_and_resize_nearest(const device ftype *image           [[buffer(0)]],
                                    const device ftype4 *boxes          [[buffer(1)]],
                                    const device int *indexes           [[buffer(2)]],
                                    device ftype *crops                 [[buffer(3)]],
                                    constant crop_and_resize_shape &s   [[buffer(4)]],
                                    uint3 gid                           [[thread_position_in_grid]]) {
    // grid protect
    int cb = gid.z / s.crop_height;
    int ch = gid.z % s.crop_height, cw = gid.y, cc = gid.x;
    if (cb >= s.boxes_count || ch >= s.crop_height || cw >= s.crop_width || cc >= s.crop_channel) return;
    
    // image index protect
    auto image_index = indexes[cb];
    if (0 > image_index || image_index >= s.image_count) return;
    
    // prepare const
    auto rect = boxes[cb];
    auto y1 = rect[0], x1 = rect[1], y2 = rect[2], x2 = rect[3];
    auto crop = crops
        + cb * s.crop_height * s.crop_width * s.crop_channel
        + ch                 * s.crop_width * s.crop_channel
        + cw                                * s.crop_channel
        + cc                                                ;
    
    // image y protect
    auto height_scale = s.crop_height > 1 ? (y2 - y1) * (s.image_height - 1) / (s.crop_height - 1) : 0;
    auto im_y = (s.crop_height > 1) ? y1 * (s.image_height - 1) + ch * height_scale : 0.5h * (y1 + y2) * (s.image_height - 1);
    if (0 > im_y || im_y > s.image_height - 1) {
        *crop = ftype(s.extrapolation);
        return;
    }
    
    // image x protect
    auto width_scale = s.crop_width > 1 ? (x2 - x1) * (s.image_width - 1) / (s.crop_width - 1) : 0;
    auto im_x = (s.crop_width > 1) ? x1 * (s.image_width - 1) + cw * width_scale : 0.5h * (x1 + x2) * (s.image_width - 1);
    if (0 > im_x || im_x > s.image_width - 1) {
        *crop = ftype(s.extrapolation);
        return;
    }
    
    // use nearest
    int nearest_y = round(im_y), nearest_x = round(im_x);
    *crop = image[image_index * s.image_height * s.image_width * s.image_channel
                  + nearest_y                  * s.image_width * s.image_channel
                  + nearest_x                                  * s.image_channel
                  + cc];
}
