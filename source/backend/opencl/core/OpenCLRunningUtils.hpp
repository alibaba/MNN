//
//  OpenCLRunningUtils.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLRunningUtils_hpp
#define OpenCLRunningUtils_hpp

#include <string>
#include <vector>

#include "Macro.h"
#include "TensorUtils.hpp"
#include "core/runtime/OpenCLRuntime.hpp"
#include "core/runtime/OpenCLWrapper.hpp"

namespace MNN {
namespace OpenCL {

inline std::vector<int> tensorShapeFormat(const Tensor *input) {
    int iN = (0 != input->buffer().dim[0].extent) ? input->buffer().dim[0].extent : 1;
    int iC = (0 != input->buffer().dim[1].extent) ? input->buffer().dim[1].extent : 1;
    int iH = (0 != input->buffer().dim[2].extent) ? input->buffer().dim[2].extent : 1;
    int iW = (0 != input->buffer().dim[3].extent) ? input->buffer().dim[3].extent : 1;

    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC) {
        iN = (0 < input->buffer().dim[0].extent) ? input->buffer().dim[0].extent : 1;
        iH = (0 < input->buffer().dim[1].extent) ? input->buffer().dim[1].extent : 1;
        iW = (0 < input->buffer().dim[2].extent) ? input->buffer().dim[2].extent : 1;
        iC = (0 < input->buffer().dim[3].extent) ? input->buffer().dim[3].extent : 1;
    }
    if (input->buffer().dimensions == 2) {
        iN = input->buffer().dim[0].extent;
        iH = 1;
        iW = 1;
        iC = input->buffer().dim[1].extent;
    }
    if (input->buffer().dimensions == 1) {
        iN = 1;
        iH = 1;
        iW = 1;
        iC = input->buffer().dim[0].extent;
    }

#ifdef LOG_VERBOSE
    MNN_PRINT("tensorShapeFormat : [%d, %d, %d, %d] \n", iN, iH, iW, iC);
#endif
    std::vector<int> shape_vec{iN, iH, iW, iC};

    return shape_vec;
}

enum OpenCLBufferFormat {
    CONV2D_FILTER    = 0,
    NHWC_BUFFER      = 1,
    ARGUMENT         = 2,
    DW_CONV2D_FILTER = 3,
    NCHW_BUFFER      = 4,
    NHWC4_BUFFER     = 5,
};

template <typename T, typename Dim>
inline void IOHW2OIHW(const T *src, T *dst, Dim O, Dim I, Dim H, Dim W) {
    for (Dim i = 0; i < I; i++) {
        for (Dim o = 0; o < O; o++) {
            for (Dim h = 0; h < H; h++) {
                for (Dim w = 0; w < W; w++) {
                    dst[o * I * H * W + i * H * W + h * W + w] = src[i * O * H * W + o * H * W + h * W + w];
                }
            }
        }
    }
};
inline cl::Buffer &openCLBuffer(const Tensor *tensor) {
    return (*(cl::Buffer *)(tensor->deviceId()));
}
inline cl::Image &openCLImage(const Tensor *tensor) {
    return (*(cl::Image *)(tensor->deviceId()));
}
void getImageShape(const std::vector<int> &shape, /* NHWC */
                   const OpenCLBufferFormat type, std::vector<size_t> *imageShape);

std::vector<uint32_t> turnLocalSize(cl::Kernel *kernel, std::vector<uint32_t> &gws, OpenCLRuntime *runtime);

void run3DKernelDefault(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                        OpenCLRuntime *runtime);

void run2DKernelDefault(const ::cl::Kernel &kernel, const uint32_t *gws, const std::vector<uint32_t> &lws,
                        OpenCLRuntime *runtime);

void runKernel2D(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                 OpenCLRuntime *runtime);
void runTurnKernelLWS2D(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                        OpenCLRuntime *runtime);

std::vector<uint32_t> localWS3DDefault(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize,
                                       OpenCLRuntime *runtime);
void copyBufferToImage(OpenCLRuntime *runtime, const cl::Buffer &buffer, const cl::Image &image, int w, int h);

} // namespace OpenCL
} // namespace MNN
#endif  /* OpenCLRunningUtils_hpp */
