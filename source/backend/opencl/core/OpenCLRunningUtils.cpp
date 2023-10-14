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

std::pair<std::vector<uint32_t>, uint32_t> localWS3DDefault(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize,
                                       OpenCLRuntime *runtime, const std::string &kernelName, const cl::Kernel &mKernel) {
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
    uint32_t min_cost = UINT_MAX;

    if(runtime->getCLTuneLevel() == Heavy) {
        while(lws[2] <= gws[2] || lws[2] <= 6) {
            lws[1] = 1;
            while(lws[1] <= gws[1] || lws[1] <= 6) {
                lws[0] = 1;
                while(lws[0] <= gws[0] || lws[0] <= 6) {
                    if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[2] <= maxWorkItemSizes[2] && lws[0]*lws[1]*lws[2] <= maxWorkGroupSize) {
                        cl::Event event;
                        std::vector<uint32_t> internalGlobalWS(3, 1);
                        for (size_t i = 0; i < gws.size(); ++i) {
                            internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                        }
                        cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                                        mKernel, cl::NullRange,
                                        cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
                                        cl::NDRange(lws[0], lws[1], lws[2]),
                                        nullptr, &event);
                        MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
                        if (res != CL_SUCCESS) {
                            MNN_PRINT("lws tune res %s\n", kernelName.c_str());
                        }
                        
                        int cost_time = (int)runtime->getCostTime(&event);
                        if(cost_time < min_cost) {
                            min_cost = cost_time;
                            lws_prefer[0] = lws[0];
                            lws_prefer[1] = lws[1];
                            lws_prefer[2] = lws[2];
                        }
                    }
                    lws[0]++;
                }
                lws[1]++;
            }
            lws[2]++;
        }
    } else if(runtime->getCLTuneLevel() == Wide) {
        while(lws[2] <= gws[2] || lws[2] <= 6) {
            lws[1] = 1;
            while(lws[1] <= gws[1] || lws[1] <= 6) {
                lws[0] = 1;
                while(lws[0] <= gws[0] || lws[0] <= 6) {
                    if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[2] <= maxWorkItemSizes[2] && lws[0]*lws[1]*lws[2] <= maxWorkGroupSize) {
                        cl::Event event;
                        std::vector<uint32_t> internalGlobalWS(3, 1);
                        for (size_t i = 0; i < gws.size(); ++i) {
                            internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                        }
                        cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                                        mKernel, cl::NullRange,
                                        cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
                                        cl::NDRange(lws[0], lws[1], lws[2]),
                                        nullptr, &event);
                        MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
                        if (res != CL_SUCCESS) {
                            MNN_PRINT("lws tune res %s\n", kernelName.c_str());
                        }
                        
                        int cost_time = (int)runtime->getCostTime(&event);
                        if(cost_time < min_cost) {
                            min_cost = cost_time;
                            lws_prefer[0] = lws[0];
                            lws_prefer[1] = lws[1];
                            lws_prefer[2] = lws[2];
                        }
                    }
                    do {
                        lws[0]++;
                    }
                    while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] > 6));//divisible powOfTwo lessThanSix
                }
                do {
                    lws[1]++;
                }
                while(((2*gws[1])%lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] > 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[2]++;
            }
            while(((2*gws[2])%lws[2] > 1) && (lws[2] & (lws[2] - 1)) != 0 && (lws[2] <= gws[2]) && (lws[2] > 6));//divisible powOfTwo lessThanSix
        }
    } else if(runtime->getCLTuneLevel() == Normal) {
        while(lws[2] <= gws[2] && lws[2] <= 6) {
            lws[1] = 1;
            while(lws[1] <= gws[1] || lws[1] <= 6) {
                lws[0] = 1;
                while(lws[0] <= gws[0] || lws[0] <= 6) {
                    if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[2] <= maxWorkItemSizes[2] && lws[0]*lws[1]*lws[2] <= maxWorkGroupSize) {
                        cl::Event event;
                        std::vector<uint32_t> internalGlobalWS(3, 1);
                        for (size_t i = 0; i < gws.size(); ++i) {
                            internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                        }
                        cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                                        mKernel, cl::NullRange,
                                        cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
                                        cl::NDRange(lws[0], lws[1], lws[2]),
                                        nullptr, &event);
                        MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
                        if (res != CL_SUCCESS) {
                            MNN_PRINT("lws tune res %s\n", kernelName.c_str());
                        }
                        
                        int cost_time = (int)runtime->getCostTime(&event);
                        if(cost_time < min_cost) {
                            min_cost = cost_time;
                            lws_prefer[0] = lws[0];
                            lws_prefer[1] = lws[1];
                            lws_prefer[2] = lws[2];
                        }
                    }
                    do {
                        lws[0]++;
                    }
                    while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] > 6));//divisible powOfTwo lessThanSix
                }
                do {
                    lws[1]++;
                }
                while(((2*gws[1])%lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] > 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[2]++;
            }
            while(((2*gws[2])%lws[2] > 1) && (lws[2] & (lws[2] - 1)) != 0 && (lws[2] <= gws[2]) && (lws[2] <= 6));//divisible powOfTwo lessThanSix
        }
    } else if(runtime->getCLTuneLevel() == Fast) {
        while(lws[2] <= gws[2] && lws[2] <= 6) {
            lws[1] = 1;
            while(lws[1] <= gws[1] && lws[1] <= 6) {
                lws[0] = 1;
                while(lws[0] <= gws[0] && lws[0] <= 6) {
                    if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[2] <= maxWorkItemSizes[2] && lws[0]*lws[1]*lws[2] <= maxWorkGroupSize) {
                        cl::Event event;
                        std::vector<uint32_t> internalGlobalWS(3, 1);
                        for (size_t i = 0; i < gws.size(); ++i) {
                            internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                        }
                        cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                                        mKernel, cl::NullRange,
                                        cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
                                        cl::NDRange(lws[0], lws[1], lws[2]),
                                        nullptr, &event);
                        MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
                        if (res != CL_SUCCESS) {
                            MNN_PRINT("lws tune res %s\n", kernelName.c_str());
                        }
                        
                        int cost_time = (int)runtime->getCostTime(&event);
                        if(cost_time < min_cost) {
                            min_cost = cost_time;
                            lws_prefer[0] = lws[0];
                            lws_prefer[1] = lws[1];
                            lws_prefer[2] = lws[2];
                        }
                    }
                    do {
                        lws[0]++;
                    }
                    while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] <= 6));//divisible powOfTwo lessThanSix
                }
                do {
                    lws[1]++;
                }
                while(((2*gws[1])%lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] <= 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[2]++;
            }
            while(((2*gws[2])%lws[2] > 1) && (lws[2] & (lws[2] - 1)) != 0 && (lws[2] <= gws[2]) && (lws[2] <= 6));//divisible powOfTwo lessThanSix
        }
    } else if(runtime->getCLTuneLevel() == None) {
        // define not tune method to choose lws
        lws_prefer[0] = 0;
        lws_prefer[1] = 0;
        lws_prefer[2] = 0;
        min_cost = 0;
    }
    
    if(runtime->getCLTuneLevel() != None) {
        cl::Event event;
        cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                        mKernel, cl::NullRange,
                        cl::NDRange(gws[0], gws[1], gws[2]),
                        cl::NullRange,
                        nullptr, &event);
        MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
        if (res != CL_SUCCESS) {
            MNN_PRINT("3D lws null res %s\n", kernelName.c_str());
        }
        
        int cost_time = (int)runtime->getCostTime(&event);
        if(cost_time < min_cost) {
            lws_prefer[0] = 0;
            lws_prefer[1] = 0;
            lws_prefer[2] = 0;
            min_cost = cost_time;
        }
    }
    
    if (tunedLws.find(info) == tunedLws.end()) {
        //printf("3dLocalWS %d Insert! gws:%d %d %d, lws:%d %d %d\n", (int)tunedLws.size(), gws[0], gws[1], gws[2], lws_prefer[0], lws_prefer[1], lws_prefer[2]);
        tunedLws.insert(std::make_pair(info, std::make_pair(lws_prefer, min_cost)));
    }

    return std::make_pair(lws_prefer, min_cost);
}

std::pair<std::vector<uint32_t>, uint32_t> localWS2DDefault(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize,
                                        OpenCLRuntime *runtime, const std::string &kernelName, const cl::Kernel &mKernel) {
    MNN_ASSERT(gws.size() == 2);
    
    auto maxWorkItemSizes = runtime->getMaxWorkItemSizes();
    MNN_ASSERT(maxWorkItemSizes.size() >= 2);
    auto& tunedLws = runtime->tunedLwsMap();
    std::pair<std::string, std::vector<uint32_t>> info = std::make_pair(kernelName, gws);
    if (tunedLws.find(info) != tunedLws.end()) {
        //printf("conv2d1x1LocalWSOpt Found! gws:%d %d lws:%d %d\n", gws[0], gws[1], tunedLws[info][0], tunedLws[info][1]);
        return tunedLws[info];
    }
    
    std::vector<uint32_t> lws(3, 1);
    std::vector<uint32_t> lws_prefer(2, 1);
    uint32_t min_cost = UINT_MAX;
    
    if(runtime->getCLTuneLevel() == Heavy) {
        while(lws[1] <= gws[1] || lws[1] <= 6) {
            lws[0] = 1;
            while(lws[0] <= gws[0] || lws[0] <= 6) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                                    mKernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                    cl::NDRange(lws[0], lws[1]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
                    if (res != CL_SUCCESS) {
                        MNN_PRINT("lws tune res %s\n", kernelName.c_str());
                    }
                    
                    int cost_time = (int)runtime->getCostTime(&event);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                lws[0]++;
            }
            lws[1]++;
        }
    } else if(runtime->getCLTuneLevel() == Wide) {
        while(lws[1] <= gws[1] || lws[1] <= 6) {
            lws[0] = 1;
            while(lws[0] <= gws[0] || lws[0] <= 6) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                                    mKernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                    cl::NDRange(lws[0], lws[1]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
                    if (res != CL_SUCCESS) {
                        MNN_PRINT("lws tune res %s\n", kernelName.c_str());
                    }
                    
                    int cost_time = (int)runtime->getCostTime(&event);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                do {
                    lws[0]++;
                }
                while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] > 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[1]++;
            }
            while(((2*gws[1])%lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] > 6));//divisible powOfTwo lessThanSix
        }
    } else if(runtime->getCLTuneLevel() == Normal) {
        while(lws[1] <= gws[1] && lws[1] <= 6) {
            lws[0] = 1;
            while(lws[0] <= gws[0] || lws[0] <= 6) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                                    mKernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                    cl::NDRange(lws[0], lws[1]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
                    if (res != CL_SUCCESS) {
                        MNN_PRINT("lws tune res %s\n", kernelName.c_str());
                    }
                    
                    int cost_time = (int)runtime->getCostTime(&event);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                do {
                    lws[0]++;
                }
                while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] > 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[1]++;
            }
            while(((2*gws[1])%lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] <= 6));//divisible powOfTwo lessThanSix
        }
    } else if(runtime->getCLTuneLevel() == Fast) {
        while(lws[1] <= gws[1] && lws[1] <= 6) {
            lws[0] = 1;
            while(lws[0] <= gws[0] && lws[0] <= 6) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                                    mKernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                    cl::NDRange(lws[0], lws[1]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
                    if (res != CL_SUCCESS) {
                        MNN_PRINT("lws tune res %s\n", kernelName.c_str());
                    }
                    
                    int cost_time = (int)runtime->getCostTime(&event);
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                do {
                    lws[0]++;
                }
                while(((2*gws[0])%lws[0] > 1) && (lws[0] & (lws[0] - 1)) != 0 && (lws[0] <= gws[0]) && (lws[0] <= 6));//divisible powOfTwo lessThanSix
            }
            do {
                lws[1]++;
            }
            while(((2*gws[1])%lws[1] > 1) && (lws[1] & (lws[1] - 1)) != 0 && (lws[1] <= gws[1]) && (lws[1] <= 6));//divisible powOfTwo lessThanSix
        }
    } else if(runtime->getCLTuneLevel() == None) {
        // define not tune method to choose lws
        lws_prefer[0] = 0;
        lws_prefer[1] = 0;
        min_cost = 0;
    }

    if(runtime->getCLTuneLevel() != None) {
        cl::Event event;
        cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                        mKernel, cl::NullRange,
                        cl::NDRange(gws[0], gws[1]),
                        cl::NullRange,
                        nullptr, &event);
        MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
        if (res != CL_SUCCESS) {
            MNN_PRINT("2D lws null res %s\n", kernelName.c_str());
        }
        
        int cost_time = (int)runtime->getCostTime(&event);
        if(cost_time < min_cost) {
            lws_prefer[0] = 0;
            lws_prefer[1] = 0;
            min_cost = cost_time;
        }
    }
    
    if (tunedLws.find(info) == tunedLws.end()) {
        //printf("2dLocalWS %d Insert! gws:%d %d, lws:%d %d\n", (int)tunedLws.size(), gws[0], gws[1], lws_prefer[0], lws_prefer[1]);
        tunedLws.insert(std::make_pair(info, std::make_pair(lws_prefer, min_cost)));
    }

    return std::make_pair(lws_prefer, min_cost);
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

    cl_int res = CL_SUCCESS;
    if(lws[0]==0 || lws[1]==0 || lws[2]==0){
        res        = runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
            cl::NullRange, nullptr, eventPtr);
    }else{
        res        = runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]),
            cl::NDRange(lws[0], lws[1], lws[2]), nullptr, eventPtr);
    }
    MNN_CHECK_CL_SUCCESS(res, "run3d");

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

    cl_int res = CL_SUCCESS;
    if(lws[0]==0 || lws[1]==0){
        res = runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NullRange, nullptr, eventPtr);

    }else{
        res = runtime->commandQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NDRange(lws[0], lws[1]), nullptr, eventPtr);
    }
    MNN_CHECK_CL_SUCCESS(res, "run2d");

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

void startRecord(OpenCLRuntime *runtime, cl_recording_qcom &recording){
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(!runtime->isUseRecordQueue()){
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("start startRecord !\n");
#endif
    cl_int res = CL_SUCCESS;
    if(runtime->isDevideOpRecord()){
        if(recording != NULL){
            clReleaseRecordingQCOM(recording);
        }
        recording = runtime->recordableQueue().NewRecordingQCOM(&res);
        MNN_CHECK_CL_SUCCESS(res, "clNewRecordingQCOM");
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end startRecord !\n");
#endif
#endif //ENABLE_OPENCL_TIME_PROFILER
}

void endRecord(OpenCLRuntime *runtime, cl_recording_qcom &recording){
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(!runtime->isUseRecordQueue()){
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("start endRecord !\n");
#endif
    if(runtime->isDevideOpRecord()){
        cl_int res = CL_SUCCESS;
        res = clEndRecordingQCOM(recording);
        MNN_CHECK_CL_SUCCESS(res, "clEndRecordingQCOM");
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end endRecord !\n");
#endif
#endif //ENABLE_OPENCL_TIME_PROFILER
}

void recordKernel2d(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                 OpenCLRuntime *runtime) {
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(!runtime->isUseRecordQueue()){
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("start recordKernel !\n");
#endif
    cl_int res = CL_SUCCESS;
    if(!runtime->isDevideOpRecord()){
        auto RecordNum = runtime->getRecordNum();
        auto maxRecordNum = runtime->getUseRecordableQueueSize();
        if(RecordNum == 0){
            cl_recording_qcom recording = runtime->recordableQueue().NewRecordingQCOM(&res);
            MNN_CHECK_CL_SUCCESS(res, "clNewRecordingQCOM");
            runtime->getRecordings()->emplace_back(recording);
        }else if(RecordNum == maxRecordNum){
            res = clEndRecordingQCOM( runtime->getRecordings()->back());
            MNN_CHECK_CL_SUCCESS(res, "clEndRecordingQCOM");
            cl_recording_qcom recording = runtime->recordableQueue().NewRecordingQCOM(&res);
            MNN_CHECK_CL_SUCCESS(res, "clNewRecordingQCOM");
            runtime->getRecordings()->emplace_back(recording);
            RecordNum = 0;
        }
        RecordNum++;
        runtime->setRecordNum(RecordNum);
    }
    
    std::vector<uint32_t> internalGlobalWS = gws;
    for (size_t i = 0; i < 2; ++i) {
        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
    }

    if(lws[0]==0 || lws[1]==0){
        res = runtime->recordableQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NullRange, nullptr, nullptr);

    }else{
        res = runtime->recordableQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]), cl::NDRange(lws[0], lws[1]), nullptr, nullptr);
    }
    MNN_CHECK_CL_SUCCESS(res, "recordKernel2d");

#ifdef LOG_VERBOSE
    MNN_PRINT("end recordKernel !\n");
#endif
#endif //ENABLE_OPENCL_TIME_PROFILER
}

void recordKernel3d(const ::cl::Kernel &kernel, const std::vector<uint32_t> &gws, const std::vector<uint32_t> &lws,
                 OpenCLRuntime *runtime) {
#if !defined(ENABLE_OPENCL_TIME_PROFILER) && defined(MNN_USE_LIB_WRAPPER)
    if(!runtime->isUseRecordQueue()){
        return;
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("start recordKernel !\n");
#endif
    cl_int res = CL_SUCCESS;
    std::vector<uint32_t> internalGlobalWS = gws;
    for (size_t i = 0; i < 3; ++i) {
        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
    }
    if(!runtime->isDevideOpRecord()){
        auto maxRecordNum = runtime->getUseRecordableQueueSize();
        auto RecordNum = runtime->getRecordNum();
        if(RecordNum == 0){
            cl_recording_qcom recording = runtime->recordableQueue().NewRecordingQCOM(&res);
            MNN_CHECK_CL_SUCCESS(res, "clNewRecordingQCOM");
            runtime->getRecordings()->emplace_back(recording);
        }else if(RecordNum == maxRecordNum){
            res = clEndRecordingQCOM( runtime->getRecordings()->back());
            MNN_CHECK_CL_SUCCESS(res, "clEndRecordingQCOM");
            cl_recording_qcom recording = runtime->recordableQueue().NewRecordingQCOM(&res);
            MNN_CHECK_CL_SUCCESS(res, "clNewRecordingQCOM");
            runtime->getRecordings()->emplace_back(recording);
            RecordNum = 0;
        }
        RecordNum++;
        runtime->setRecordNum(RecordNum);
    }

    if(lws[0]==0 || lws[1]==0 || lws[2]==0){
        res = runtime->recordableQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]), cl::NullRange, nullptr, nullptr);

    }else{
        res = runtime->recordableQueue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(internalGlobalWS[0], internalGlobalWS[1], internalGlobalWS[2]), cl::NDRange(lws[0], lws[1], lws[2]), nullptr, nullptr);
    }
    MNN_CHECK_CL_SUCCESS(res, "recordKernel3d");
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end recordKernel !\n");
#endif
#endif //ENABLE_OPENCL_TIME_PROFILER
}

} // namespace OpenCL
} // namespace MNN
