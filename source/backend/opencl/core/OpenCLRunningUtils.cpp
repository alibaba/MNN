//
//  OpenCLRunningUtils.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include "core/Macro.h"

namespace MNN {
namespace OpenCL {

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
    } else if(type == CONV2D1x1_OPT_FILTER){
        (*imageShape).push_back(UP_DIV(shape[1], 4));
        (*imageShape).push_back(shape[2] * shape[3] * shape[0]);
    }else {
        MNN_PRINT("type not supported !!! \n");
    }
}

std::vector<uint32_t> localWS3DDefault(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize,
                                       OpenCLRuntime *runtime, std::string &kernelName, cl::Kernel &mKernel) {
#ifdef MNN_OPENCL_LWS_TUNE
    MNN_ASSERT(gws.size() == 3);
    
    auto maxWorkItemSizes = runtime->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 3);
    auto& tunedLws = runtime->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(kernelName, gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("conv2d1x1LocalWSOpt Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
        return tunedLws[info];
    }
    
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(4, 1);
    int min_cost = INT_MAX;

    while(lws[2] <= gws[2]) {
        lws[1] = 1;
        while(lws[1] <= gws[1]) {
            lws[0] = 1;
            while(lws[0] <= gws[0]) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[2] <= maxWorkItemSizes[2] && lws[0]*lws[1]*lws[2] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(3, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int error = runtime->commandQueue().enqueueNDRangeKernel(
                                    mKernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
                                    cl::NDRange(lws[0], lws[1], lws[2]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(error);
                    if (error != CL_SUCCESS) {
                        printf("%s\n", kernelName.c_str());
                    }
                    
                    int cost_time = (int)runtime->getCostTime(&event);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                        lws_prefer[2] = lws[2];
                    }
                }
                lws[0] *= 2;
            }
            lws[1] *= 2;
        }
        lws[2] *= 2;
    }
    
    if (tunedLws.find(info) == tunedLws.end()) {
        //printf("conv2d1x1LocalWSOpt %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
        tunedLws.insert(std::make_pair(info, lws_prefer));
    }

    return lws_prefer;
#else
    
    std::vector<uint32_t> lws(4, 0);
    auto maxWorkItemSizes = runtime->getMaxWorkItemSizes();
    GpuType gpuType             = runtime->getGpuType();
    uint32_t deviceComputeUnits = runtime->deviceComputeUnits();
    int coreNum   = deviceComputeUnits;
    for (int i = 0, totalSizeNow = 1; i < gws.size(); ++i) {
        int remain = gws[i] % coreNum, groupSize = gws[i] / coreNum;
        if (remain == 0) {
            lws[i] = groupSize;
        } else {
            while(groupSize) {
                int remain = gws[i] % groupSize;
                if (remain == 0 && (i > 0 || groupSize <= maxWorkGroupSize)) {
                    lws[i] = groupSize;
                    break;
                }
                --groupSize;
            }
        }
        int limit = std::min<uint32_t>(maxWorkGroupSize / totalSizeNow, maxWorkItemSizes[i]);
        lws[i] = std::max<uint32_t>(std::min<uint32_t>(lws[i], limit), 1);
        totalSizeNow *= lws[i];
    }
    return lws;
#endif
}

void run3DKernelDefault(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                        OpenCLRuntime *runtime, cl::Event* eventPtr) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start run3DKernelDefault !\n");
#endif

    MNN_ASSERT(lws.size() >= 3);
    std::vector<uint32_t> internalGlobalWS = gws;
    for (size_t i = 0; i < 3; ++i) {
        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
    }

    cl_int error = CL_SUCCESS;
    if(eventPtr == nullptr){
        error        = runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
            cl::NDRange(lws[0], lws[1], lws[2]));

    }else{
        error        = runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
            cl::NDRange(lws[0], lws[1], lws[2]), nullptr, eventPtr);
    }
    MNN_CHECK_CL_SUCCESS(error);

    unsigned int num_flush = runtime->getQueueNum();
    if(runtime->getGpuType() != GpuType::ADRENO) {
        if(num_flush % 2 == 0) {
            runtime->commandQueue().flush();
        }
    }
    else {
        if(num_flush % 10 == 0) {
            runtime->commandQueue().flush();
        }
    }
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end run3DKernelDefault !\n");
#endif
}

void runKernel2D(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                 OpenCLRuntime *runtime,  cl::Event* eventPtr) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start runKernel2D !\n");
#endif

    std::vector<uint32_t> internalGlobalWS = gws;
    for (size_t i = 0; i < 2; ++i) {
        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
    }

    cl_int error = CL_SUCCESS;
    if(eventPtr == nullptr){
        error        = runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NDRange(lws[0], lws[1]));

    }else{
        error        = runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NDRange(lws[0], lws[1]), nullptr, eventPtr);
    }
    MNN_CHECK_CL_SUCCESS(error);

    unsigned int num_flush = runtime->getQueueNum();
    if(runtime->getGpuType() != GpuType::ADRENO) {
        if(num_flush % 2 == 0) {
            runtime->commandQueue().flush();
        }
    }
    else {
        if(num_flush % 10 == 0) {
            runtime->commandQueue().flush();
        }
    }

    
#ifdef LOG_VERBOSE
    MNN_PRINT("end runKernel2D !\n");
#endif
}

void run2DKernelDefault(const cl::Kernel &kernel, const uint32_t *gws, const std::vector<uint32_t> &lws,
                        OpenCLRuntime *runtime) {

    const std::vector<uint32_t> &params = lws;
    MNN_ASSERT(params.size() == 3);
    std::vector<uint32_t> internalGlobalWS(gws, gws + 2);
    for (size_t i = 0; i < 2; ++i) {
        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, params[i]));
    }

    uint32_t block_size       = params[2] == 0 ? internalGlobalWS[1] : params[2];
    const uint32_t num_blocks = UP_DIV(internalGlobalWS[1], block_size);
    cl_int error = CL_SUCCESS;
    
#ifdef ENABLE_OPENCL_TIME_PROFILER
    int idx = 0;
#endif
    for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t gws1 = block_size;
    #ifdef ENABLE_OPENCL_TIME_PROFILER
        cl::Event event;
        error |= runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NDRange(0, i * block_size),
            cl::NDRange(internalGlobalWS[0], gws1),
            cl::NDRange(params[0], params[1]), nullptr, &event);
        int costTime = (int)runtime->getCostTime(&event);
        MNN_PRINT("kernel cost:%d    us run2DKernelDefault%d\n",costTime, idx++);
    #else
        error |= runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NDRange(0, i * block_size),
            cl::NDRange(internalGlobalWS[0], gws1),
            cl::NDRange(params[0], params[1]));
    #endif
    }
    MNN_CHECK_CL_SUCCESS(error);

    unsigned int num_flush = runtime->getQueueNum();
    if(runtime->getGpuType() != GpuType::ADRENO) {
        if(num_flush % 2 == 0) {
            runtime->commandQueue().flush();
        }
    }
    else {
        if(num_flush % 10 == 0) {
            runtime->commandQueue().flush();
        }
    }
    
}
void copyBufferToImage(OpenCLRuntime *runtime, const cl::Buffer &buffer, const cl::Image &image, int w, int h) {
    std::set<std::string> buildOptions;
    if(runtime->isWeightCpuTransHalf() == false) {
        buildOptions.emplace("-DBUFFER_INP_FP32");
    }
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
