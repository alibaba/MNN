//
//  ConvertUtils.hpp
//  MNN
//
//  Created by MNN on 2020/04/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvertUtils_hpp
#define ConvertUtils_hpp
#include "geometry/GeometryComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
class ConvertUtils {
public:
    static bool compute(Tensor* input, Tensor* output, CommandBuffer& res);
    // numpy broadcast like: [3, 4] -> [2, 3, 4]
    // forward = true: [4] -> [4, 3, 2]
    static void broadcastto(Tensor* input, Tensor* output, bool forward = false);
};

/**
 if coordinate_transformation_mode is "half_pixel",
 x_original = (x_resized + 0.5) / scale - 0.5,

 if coordinate_transformation_mode is "pytorch_half_pixel",
 x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0,

 if coordinate_transformation_mode is "align_corners",
 x_original = x_resized * (length_original - 1) / (length_resized - 1),

 if coordinate_transformation_mode is "asymmetric",
 x_original = x_resized / scale,

 if coordinate_transformation_mode is "tf_half_pixel_for_nn",
 x_original = (x_resized + 0.5) / scale,

 if coordinate_transformation_mode is "tf_crop_and_resize",
 x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1).
 */
struct InterpInfo {
    float depthScale;
    float heightScale;
    float widthScale;
    float widthOffset = 0.0f;
    float heightOffset = 0.0f;
    float depthOffset = 0.0f;
};

static void _ConverterInterp(const Interp* resize, InterpInfo* dstInfo, int inW, int inH, int inD, int outW, int outH, int outD, bool computeScale = true) {
    switch (resize->ctm()) {
        case CoordinateTransformationMode_NotSet:
        {
            // For compability, old model's nearest don't support halfpixels
            if (resize->halfPixelCenters() && resize->resizeType() != 1) {
                if (computeScale) {
                    dstInfo->depthScale  = (float)(inD) / (float)(outD);
                    dstInfo->heightScale = (float)(inH) / (float)(outH);
                    dstInfo->widthScale  = (float)(inW) / (float)(outW);
                }
                dstInfo->widthOffset = 0.5f * dstInfo->widthScale - 0.5f;
                dstInfo->heightOffset = 0.5f * dstInfo->heightScale - 0.5f;
                dstInfo->depthOffset = 0.5f * dstInfo->depthScale - 0.5f;
            } else if (resize->alignCorners()) {
                if (computeScale) {
                    if (outD == 1) {
                        dstInfo->depthScale = 0.0f;
                    } else {
                        dstInfo->depthScale = (float)(inD - 1) / (float)(outD - 1);
                    }
                    if (outH == 1) {
                        dstInfo->heightScale = 0.0f;
                    } else {
                        dstInfo->heightScale = (float)(inH - 1) / (float)(outH - 1);
                    }
                    if (outW == 1) {
                        dstInfo->widthScale = 0.0f;
                    } else {
                        dstInfo->widthScale  = (float)(inW - 1) / (float)(outW - 1);
                    }
                }
            } else if (computeScale) {
                dstInfo->depthScale = (float)(inD) / (float)(outD);
                dstInfo->heightScale = (float)(inH) / (float)(outH);
                dstInfo->widthScale  = (float)(inW) / (float)(outW);
            }
            break;
        }
        case CoordinateTransformationMode_AlignCorners:
        {
            if (outD == 1) {
                dstInfo->depthScale = 0.0f;
            } else {
                dstInfo->depthScale = (float)(inD - 1) / (float)(outD - 1);
            }
            if (outH == 1) {
                dstInfo->heightScale = 0.0f;
            } else {
                dstInfo->heightScale = (float)(inH - 1) / (float)(outH - 1);
            }
            if (outW == 1) {
                dstInfo->widthScale = 0.0f;
            } else {
                dstInfo->widthScale  = (float)(inW - 1) / (float)(outW - 1);
            }
            break;
        }
        case CoordinateTransformationMode_HalfPixels:
        {
            if (computeScale) {
                dstInfo->depthScale = (float)(inD) / (float)(outD);
                dstInfo->heightScale = (float)(inH) / (float)(outH);
                dstInfo->widthScale  = (float)(inW) / (float)(outW);
            }
            dstInfo->widthOffset = 0.5f * dstInfo->widthScale - 0.5f;
            dstInfo->heightOffset = 0.5f * dstInfo->heightScale - 0.5f;
            dstInfo->depthOffset = 0.5f * dstInfo->depthScale - 0.5f;
            break;
        }
        case CoordinateTransformationMode_PytorchHalfPixels:
        {
            if (outD > 1) {
                if (computeScale) {
                    dstInfo->depthScale = (float)inD / (float)outD;
                }
                dstInfo->depthScale = 0.5f * dstInfo->depthScale - 0.5f;
            } else {
                if (computeScale) {
                    dstInfo->depthScale = 0.0f;
                }
            }
            if (outH > 1) {
                if (computeScale) {
                    dstInfo->heightScale = (float)inH / (float)outH;
                }
                dstInfo->heightOffset = 0.5f * dstInfo->heightScale - 0.5f;
            } else {
                if (computeScale) {
                    dstInfo->heightScale = 0.0f;
                }
            }
            if (outW > 1) {
                if (computeScale) {
                    dstInfo->widthScale = (float)inW / (float)outW;
                }
                dstInfo->widthOffset = 0.5f * dstInfo->widthScale - 0.5f;
            } else {
                if (computeScale) {
                    dstInfo->widthScale = 0.0f;
                }
            }
            break;
        }
        case CoordinateTransformationMode_Asymmetric:
        {
            if (computeScale) {
                dstInfo->depthScale = (float)(inD) / (float)(outD);
                dstInfo->heightScale = (float)(inH) / (float)(outH);
                dstInfo->widthScale  = (float)(inW) / (float)(outW);
            }
            break;
        }
        case CoordinateTransformationMode_TensorflowHalfPixels:
        {
            if (computeScale) {
                dstInfo->depthScale = (float)(inD) / (float)(outD);
                dstInfo->heightScale = (float)(inH) / (float)(outH);
                dstInfo->widthScale  = (float)(inW) / (float)(outW);
            }
            dstInfo->widthOffset = 0.5f * dstInfo->widthScale;
            dstInfo->heightOffset = 0.5f * dstInfo->heightScale;
            dstInfo->depthOffset = 0.5f * dstInfo->depthScale;
            break;
        }
        case CoordinateTransformationMode_TensorflowCropAndResize:
        {
            //FIXME: Not support now
            MNN_ERROR("Don't support CoordinateTransformationMode_TensorflowCropAndResize currently\n");
            break;
        }
        default:
            break;
    }
}

} // namespace MNN

#endif
