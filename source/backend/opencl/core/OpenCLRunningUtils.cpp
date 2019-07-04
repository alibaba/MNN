//
//  OpenCLRunningUtils.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/OpenCLRunningUtils.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include "Macro.h"

namespace MNN {
namespace OpenCL {

std::vector<uint32_t> turnLocalSize(cl::Kernel *kernel, std::vector<uint32_t> &gws, OpenCLRuntime *runtime) {
    uint32_t maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(*kernel));

    int64_t minExecTime                    = std::numeric_limits<int64_t>::max();
    std::vector<uint32_t> optimizedLocalWS = {1, 1, 1};
    const int xEnd                         = 32;
    const int yEnd                         = 32;

    for (uint32_t y = 1; y <= yEnd; ++y) {
        for (uint32_t x = 1; x <= xEnd; ++x) {
            cl::NDRange LocalWorkSize = cl::NDRange(x, y);

            const bool invalid_lws = (x * y > maxWorkGroupSize) || (x == 1 && y == 1);

            if (invalid_lws) {
                continue;
            }

            std::vector<uint32_t> roundGWS = gws;
            for (size_t i = 0; i < 2; ++i) {
                MNN_ASSERT(LocalWorkSize[i] != 0);
                roundGWS[i] = ROUND_UP(gws[i], LocalWorkSize[i]);
            }

            int64_t cost_time = 0;
            for (int i = 0; i < 3; i++) {
                cl::Event event;
                cl_int error            = CL_SUCCESS;
                const int64_t startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                error                   = runtime->commandQueue().enqueueNDRangeKernel(
                    *kernel, cl::NullRange, cl::NDRange(roundGWS[0], roundGWS[1]),
                    cl::NDRange(LocalWorkSize[0], LocalWorkSize[1]), nullptr, &event);

                event.wait();
                const int64_t endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                cost_time += (endTime - startTime);
            }

            if (cost_time < minExecTime) {
                minExecTime      = cost_time;
                optimizedLocalWS = {x, y};
            }
        }
    }

    MNN_PRINT("best lws : [%d, %d] \n", optimizedLocalWS[0], optimizedLocalWS[1]);
    return optimizedLocalWS;
}

void getImageShape(const std::vector<int> &shape, const OpenCLBufferFormat type, std::vector<size_t> *imageShape) {
    MNN_ASSERT(imageShape != nullptr);
    if (type == CONV2D_FILTER) {
        (*imageShape).push_back(shape[1]);
        (*imageShape).push_back(shape[2] * shape[3] * UP_DIV(shape[0], 4));
    } else if (type == DW_CONV2D_FILTER) {
        (*imageShape).push_back(shape[0] * shape[2] * shape[3]);
        (*imageShape).push_back(UP_DIV(shape[1], 4));
    } else if (type == NHWC_BUFFER || type == NCHW_BUFFER) {
        (*imageShape).push_back(UP_DIV(shape[3], 4) * shape[2]);
        (*imageShape).push_back(shape[0] * shape[1]);
    } else if (type == ARGUMENT) {
        if (shape.size() == 4) {
            (*imageShape).push_back(UP_DIV(shape[3], 4));
            (*imageShape).push_back(1);
        } else {
            (*imageShape).push_back(UP_DIV(shape[0], 4));
            (*imageShape).push_back(1);
        }
    } else {
        MNN_PRINT("type not supported !!! \n");
    }
}

std::vector<uint32_t> localWS3DDefault(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize,
                                       OpenCLRuntime *runtime) {
    std::vector<uint32_t> lws(4, 0);
    GpuType gpuType             = runtime->getGpuType();
    uint32_t deviceComputeUnits = runtime->deviceComputeUnits();
    if (gpuType == GpuType::ADRENO) {
        int coreNum   = deviceComputeUnits;
        int remain    = gws[0] % coreNum;
        int groupSize = gws[0] / coreNum;
        if (remain == 0) {
            lws[0] = groupSize;
        } else {
            while (groupSize) {
                int remain = gws[0] % groupSize;
                if (remain == 0 && groupSize <= maxWorkGroupSize) {
                    lws[0] = groupSize;
                    break;
                }
                groupSize--;
            }
        }
        lws[0] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize, lws[0]), 1);

        remain    = gws[1] % coreNum;
        groupSize = gws[1] / coreNum;
        if (remain == 0) {
            lws[1] = groupSize;
        } else {
            while (groupSize) {
                int remain = gws[1] % groupSize;
                if (remain == 0) {
                    lws[1] = groupSize;
                    break;
                }
                groupSize--;
            }
        }
        lws[1] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize / lws[0], lws[1]), 1);

        remain    = gws[2] % coreNum;
        groupSize = gws[2] / coreNum;
        if (remain == 0) {
            lws[2] = groupSize;
        } else {
            while (groupSize) {
                int remain = gws[2] % groupSize;
                if (remain == 0) {
                    lws[2] = groupSize;
                    break;
                }
                groupSize--;
            }
        }

        lws[2] = std::max<uint32_t>(std::min<uint32_t>(maxWorkGroupSize / (lws[0] * lws[1]), lws[2]), 1);
    } else {
        lws[0] = deviceComputeUnits * 2;
        lws[1] = 4;
        lws[2] = 1;
    }
    return lws;
}

void runTurnKernelLWS2D(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                        OpenCLRuntime *runtime) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start runTurnKernelLWS2D !\n");
#endif

    std::vector<uint32_t> roundGWS = gws;
    for (size_t i = 0; i < 2; ++i) {
        MNN_ASSERT(lws[i] != 0);
        roundGWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
    }

    cl::Event event;
    cl_int error = CL_SUCCESS;
    error = runtime->commandQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(roundGWS[0], roundGWS[1]),
                                                         cl::NDRange(lws[0], lws[1]), nullptr, &event);
    MNN_CHECK_CL_SUCCESS(error);

#ifdef LOG_VERBOSE
    MNN_PRINT("end runTurnKernelLWS2D !\n");
#endif
}

void run3DKernelDefault(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                        OpenCLRuntime *runtime) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start run3DKernelDefault !\n");
#endif

    MNN_ASSERT(lws.size() >= 3);
    std::vector<uint32_t> internalGlobalWS = gws;
    for (size_t i = 0; i < 3; ++i) {
        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
    }

    cl_int error = CL_SUCCESS;
    error        = runtime->commandQueue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
        cl::NDRange(lws[0], lws[1], lws[2]));

    MNN_CHECK_CL_SUCCESS(error);

#ifdef LOG_VERBOSE
    MNN_PRINT("end run3DKernelDefault !\n");
#endif
}

void runKernel2D(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                 OpenCLRuntime *runtime) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start run3DKernelDefault !\n");
#endif

    std::vector<uint32_t> internalGlobalWS = gws;
    for (size_t i = 0; i < 2; ++i) {
        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
    }

    cl_int error = CL_SUCCESS;
    error        = runtime->commandQueue().enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NDRange(lws[0], lws[1]));

    MNN_CHECK_CL_SUCCESS(error);

#ifdef LOG_VERBOSE
    MNN_PRINT("end run3DKernelDefault !\n");
#endif
}

void run2DKernelDefault(const cl::Kernel &kernel, const uint32_t *gws, const std::vector<uint32_t> &lws,
                        OpenCLRuntime *runtime) {
    cl::Event event;
    const std::vector<uint32_t> &params = lws;
    MNN_ASSERT(params.size() == 3);
    std::vector<uint32_t> internalGlobalWS(gws, gws + 2);
    for (size_t i = 0; i < 2; ++i) {
        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, params[i]));
    }

    uint32_t block_size       = params[2] == 0 ? internalGlobalWS[1] : params[2];
    const uint32_t num_blocks = UP_DIV(internalGlobalWS[1], block_size);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t gws1 = block_size;
        MNN_CHECK_CL_SUCCESS(runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NDRange(0, i * block_size), cl::NDRange(internalGlobalWS[0], gws1),
            cl::NDRange(params[0], params[1]), nullptr, &event));
    }
}
void copyBufferToImage(OpenCLRuntime *runtime, const cl::Buffer &buffer, const cl::Image &image, int w, int h) {
    std::set<std::string> buildOptions;
    auto kernel = runtime->buildKernel("copy_buffer_to_image2d", "copy_buffer_to_image2d", buildOptions);
    auto status = kernel.setArg(0, buffer);
    MNN_ASSERT(status == CL_SUCCESS);
    status = kernel.setArg(1, image);
    MNN_ASSERT(status == CL_SUCCESS);
    status = kernel.setArg(2, w);
    MNN_ASSERT(status == CL_SUCCESS);
    status = kernel.setArg(3, h);
    MNN_ASSERT(status == CL_SUCCESS);
    auto comandQueue = runtime->commandQueue();
    comandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(w, h, 1));
}

} // namespace OpenCL
} // namespace MNN
