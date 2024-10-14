//
//  RasterExecution.cpp
//  MNN
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/execution/image/RasterExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {


RasterExecution::RasterExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = (OpenCLBackend *)backend;
    //nothing to do
}

ErrorCode RasterExecution::onEncode(const std::vector<Tensor *> &____inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RasterExecution onResize !\n");
#endif
    mTempInput.clear();
    mTempOutput = nullptr;
    MNN_ASSERT(outputs.size() == 1);
    auto output = outputs[0];
    OpCommonUtils::rasterInputReset(____inputs, outputs[0]);

    auto des = TensorUtils::getDescribe(output);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero = !TensorUtils::regionIsFull(output);
    auto regionNum = des->regions.size();
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    mFast = false;
    if (outputDes->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        mFast = true;
        for (int i=0; i< des->regions.size(); ++i) {
            auto& slice = des->regions[i];
            if (TensorUtils::getDescribe(slice.origin)->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                mFast = false;
                break;
            }
            if (!OpCommonUtils::canBlitFast(slice, output)) {
                mFast = false;
                break;
            }
        }
    }

    if(mFast)
    {
        mUnits.resize(regionNum);
        int kernel_idx = 0;
        
        if(mNeedZero)
        {
            mUnits.resize(regionNum + 1);
            auto outputShape    = tensorShapeFormat(output);
            int region[] = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[1], outputShape[2]};//nhwc
            Unit &unit          = mUnits[kernel_idx++];
            unit.kernel         = runtime->buildKernel("raster", "image_set_zero", {}, output, output);
            unit.localWorkSize  = {8, 8};
            unit.globalWorkSize = {(uint32_t)UP_DIV((region[1] * region[3]), 16)*16,
                                   (uint32_t)UP_DIV((region[0] * region[2]), 16)*16};

            int global_dim0 = region[1] * region[3];
            int global_dim1 = region[0] * region[2];

            uint32_t idx   = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, global_dim0);
            ret |= unit.kernel->get().setArg(idx++, global_dim1);
            ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
            if(ret != CL_SUCCESS)
            {
                MNN_PRINT("setArg err %d\n", (int)ret);
            }
            mOpenCLBackend->recordKernel2d(unit.kernel,
                {(uint32_t)UP_DIV((region[1] * region[3]), 16)*16,
                (uint32_t)UP_DIV((region[0] * region[2]), 16)*16},
                {8, 8});
        }
        
        // image raster
        for (auto& slice : des->regions)
        {
            Tensor::InsideDescribe::Region C4Region;
            OpCommonUtils::turnToPackRegion(slice, C4Region, output, 4);

            Unit &unit          = mUnits[kernel_idx++];
            unit.kernel         = runtime->buildKernel("raster", "raster_image", {}, output, output);

            const std::vector<uint32_t> gws =  {(uint32_t)C4Region.size[2],
                                                    (uint32_t)C4Region.size[1],
                                                    (uint32_t)C4Region.size[0]};
            uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

            auto outputShape    = tensorShapeFormat(output);
            auto sliceShape    = tensorShapeFormat(slice.origin);

            uint32_t idx   = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, gws[0]);
            ret |= unit.kernel->get().setArg(idx++, gws[1]);
            ret |= unit.kernel->get().setArg(idx++, gws[2]);
            ret |= unit.kernel->get().setArg(idx++, openCLImage(slice.origin));
            ret |= unit.kernel->get().setArg(idx++, C4Region.src.offset);
            ret |= unit.kernel->get().setArg(idx++, C4Region.src.stride[0]);
            ret |= unit.kernel->get().setArg(idx++, C4Region.src.stride[1]);
            ret |= unit.kernel->get().setArg(idx++, C4Region.src.stride[2]);
            ret |= unit.kernel->get().setArg(idx++, sliceShape[1]);
            ret |= unit.kernel->get().setArg(idx++, sliceShape[2]);
            ret |= unit.kernel->get().setArg(idx++, sliceShape[3]);
            ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
            ret |= unit.kernel->get().setArg(idx++, C4Region.dst.offset);
            ret |= unit.kernel->get().setArg(idx++, C4Region.dst.stride[0]);
            ret |= unit.kernel->get().setArg(idx++, C4Region.dst.stride[1]);
            ret |= unit.kernel->get().setArg(idx++, C4Region.dst.stride[2]);
            ret |= unit.kernel->get().setArg(idx++, outputShape[1]);
            ret |= unit.kernel->get().setArg(idx++, outputShape[2]);
            ret |= unit.kernel->get().setArg(idx++, outputShape[3]);
            if(ret != CL_SUCCESS)
            {
                MNN_PRINT("setArg err %d\n", (int)ret);
            }
            std::string name = "rasterImage";
            const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
            
            unit.localWorkSize = {lws[0], lws[1], lws[2]};
            
            unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
                                   ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
                                   ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]))};
            mOpenCLBackend->recordKernel3d(unit.kernel, gws, lws);
        }
        return NO_ERROR;
    }
    
    bool cancombine = CanCombine(outputs);
    // Alloc Temp buffer
    auto bufferPool     = ((OpenCLBackend *)backend())->getBufferPool();
    auto bufferUnitSize = runtime->isSupportedFP16() ? sizeof(half_float::half) : sizeof(float);
    for(int i=0; i< regionNum; ++i)
    {
        auto origin = des->regions[i].origin;
        if(mTempInput.find(origin) != mTempInput.end())
        {
            continue;
        }

        auto buffer = bufferPool->alloc(origin->elementSize()*bufferUnitSize);
        mTempInput.insert(std::make_pair(origin, buffer));
    }
    mTempOutput         = bufferPool->alloc(output->elementSize() * bufferUnitSize);

    for(auto& iter : mTempInput)
    {
        bufferPool->recycle(iter.second);
    }
    bufferPool->recycle(mTempOutput);
    
    auto originNum = mTempInput.size();
    if(cancombine){
        regionNum = 1;
    }
    mUnits.resize(regionNum + originNum + 1);
    
    int kernel_idx = 0;
    if(mNeedZero)
    {
        mUnits.resize(regionNum + originNum + 2);
        auto outputShape    = tensorShapeFormat(output);
        int region[] = {outputShape[0], outputShape[3], outputShape[1], outputShape[2]};//nhwc
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster", "buffer_set_zero", {}, output, output);

        std::vector<uint32_t> gws = {(uint32_t)(region[2] * region[3]),
                                     (uint32_t)(region[0] * region[1])};
        
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, gws[0]);
        ret |= unit.kernel->get().setArg(idx++, gws[1]);
        ret |= unit.kernel->get().setArg(idx++, *mTempOutput);
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        std::string kernelName = "raster_buffer_set_zero";
        std::vector<uint32_t> lws = localWS2DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;

        unit.localWorkSize = {lws[0], lws[1]};
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
            ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]))};
        
        mOpenCLBackend->recordKernel2d(unit.kernel, gws, lws);
    }

    //image to buffer
    for(auto& iter : mTempInput)
    {
        Tensor* origin = iter.first;
        std::vector<int> regionShape = tensorShapeFormat(origin);
        int inputWH[]      = {regionShape[2], regionShape[1]};
        int region[]       = {regionShape[0], UP_DIV(regionShape[3], 4), regionShape[1], regionShape[2]};
                
        Unit &unit          = mUnits[kernel_idx++];
        if(TensorUtils::getDescribe(origin)->dimensionFormat == MNN_DATA_FORMAT_NHWC)// Image to nhwc buffer
        {
            unit.kernel         = runtime->buildKernel("buffer_to_image", "image_to_nhwc_buffer", {}, origin, origin);
        }
        else //Image to nchw buffer
        {
            unit.kernel         = runtime->buildKernel("buffer_to_image", "image_to_nchw_buffer", {}, origin, origin);
        }

        std::vector<uint32_t> gws = {(uint32_t)(region[3] * region[1]),
                                     (uint32_t)(region[2] * region[0])};
        //MNN_CHECK_CL_SUCCESS
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, gws[0]);
        ret |= unit.kernel->get().setArg(idx++, gws[1]);
        ret |= unit.kernel->get().setArg(idx++, *(iter.second));
        ret |= unit.kernel->get().setArg(idx++, inputWH[1]);
        ret |= unit.kernel->get().setArg(idx++, inputWH[0]);
        ret |= unit.kernel->get().setArg(idx++, regionShape[3]);
        ret |= unit.kernel->get().setArg(idx++, openCLImage(origin));
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        std::string kernelName = "raster_image_to_buffer";
        std::vector<uint32_t> lws = localWS2DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;

        unit.localWorkSize = {lws[0], lws[1]};
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
            ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]))};
        mOpenCLBackend->recordKernel2d(unit.kernel, gws, lws);
    }
    
    // buffer raster
    if(cancombine){
        auto regions = des->regions;
        auto slice = regions[0];
        int nums = regions.size();
        int src_offset = regions[1].src.offset - slice.src.offset;
        int dst_offset = regions[1].dst.offset - slice.dst.offset;
        
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster", "raster_buffer_combine", {}, output, output);
        
        unit.globalWorkSize = {(uint32_t)slice.size[2] * nums,
            (uint32_t)slice.size[1],
            (uint32_t)slice.size[0]};
        
        const std::vector<uint32_t> gws =  {(uint32_t)slice.size[2] * nums,
            (uint32_t)slice.size[1],
            (uint32_t)slice.size[0]};
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, gws[0]);
        ret |= unit.kernel->get().setArg(idx++, gws[1]);
        ret |= unit.kernel->get().setArg(idx++, gws[2]);
        ret |= unit.kernel->get().setArg(idx++, *(mTempInput[slice.origin]));
        ret |= unit.kernel->get().setArg(idx++, slice.src.offset);
        ret |= unit.kernel->get().setArg(idx++, src_offset);
        ret |= unit.kernel->get().setArg(idx++, slice.src.stride[0]);
        ret |= unit.kernel->get().setArg(idx++, slice.src.stride[1]);
        ret |= unit.kernel->get().setArg(idx++, slice.src.stride[2]);
        ret |= unit.kernel->get().setArg(idx++, *mTempOutput);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.offset);
        ret |= unit.kernel->get().setArg(idx++, dst_offset);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[0]);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[1]);
        ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[2]);
        ret |= unit.kernel->get().setArg(idx++, slice.size[2]);
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        std::string name = "rasterBuffer";
        const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
        
        unit.localWorkSize = {lws[0], lws[1], lws[2]};
        
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
            ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
            ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]))};
        mOpenCLBackend->recordKernel3d(unit.kernel, gws, lws);
    }else{
        for (auto& slice : des->regions)
        {
            Unit &unit          = mUnits[kernel_idx++];
            unit.kernel         = runtime->buildKernel("raster", "raster_buffer", {}, output, output);
            
            unit.globalWorkSize = {(uint32_t)slice.size[2],
                (uint32_t)slice.size[1],
                (uint32_t)slice.size[0]};
            
            const std::vector<uint32_t> gws =  {(uint32_t)slice.size[2],
                (uint32_t)slice.size[1],
                (uint32_t)slice.size[0]};
            uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
            
            uint32_t idx   = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel->get().setArg(idx++, gws[0]);
            ret |= unit.kernel->get().setArg(idx++, gws[1]);
            ret |= unit.kernel->get().setArg(idx++, gws[2]);
            ret |= unit.kernel->get().setArg(idx++, *(mTempInput[slice.origin]));
            ret |= unit.kernel->get().setArg(idx++, slice.src.offset);
            ret |= unit.kernel->get().setArg(idx++, slice.src.stride[0]);
            ret |= unit.kernel->get().setArg(idx++, slice.src.stride[1]);
            ret |= unit.kernel->get().setArg(idx++, slice.src.stride[2]);
            ret |= unit.kernel->get().setArg(idx++, *mTempOutput);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.offset);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[0]);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[1]);
            ret |= unit.kernel->get().setArg(idx++, slice.dst.stride[2]);
            if(ret != CL_SUCCESS)
            {
                MNN_PRINT("setArg err %d\n", (int)ret);
            }
            
            std::string name = "rasterBuffer";
            const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
            
            unit.localWorkSize = {lws[0], lws[1], lws[2]};
            
            unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
                ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
                ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]))};
            mOpenCLBackend->recordKernel3d(unit.kernel, gws, lws);
        }
    }
    
    //buffer to image
    {
        auto outputShape    = tensorShapeFormat(output);
        int wh[]     = {outputShape[2], outputShape[1]};
        int region[] = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[1], outputShape[2]};

        Unit &unit          = mUnits[kernel_idx++];
        if(outputDes->dimensionFormat == MNN_DATA_FORMAT_NHWC)//nhwc buffer to Image
        {
            unit.kernel         = runtime->buildKernel("buffer_to_image", "nhwc_buffer_to_image", {}, output, output);
        }
        else //nchw buffer to Image
        {
            unit.kernel         = runtime->buildKernel("buffer_to_image", "nchw_buffer_to_image", {}, output, output);
        }
        
        std::vector<uint32_t> gws = {(uint32_t)(region[3] * region[1]),
                                     (uint32_t)(region[2] * region[0])};
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel->get().setArg(idx++, gws[0]);
        ret |= unit.kernel->get().setArg(idx++, gws[1]);
        ret |= unit.kernel->get().setArg(idx++, *mTempOutput);
        ret |= unit.kernel->get().setArg(idx++, wh[1]);
        ret |= unit.kernel->get().setArg(idx++, wh[0]);
        ret |= unit.kernel->get().setArg(idx++, outputShape[3]);
        ret |= unit.kernel->get().setArg(idx++, openCLImage(output));
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
        
        std::string kernelName = "raster_buffer_to_image";
        std::vector<uint32_t> lws = localWS2DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), kernelName, unit.kernel).first;

        unit.localWorkSize = {lws[0], lws[1]};
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
            ROUND_UP(gws[1], std::max((uint32_t)1, lws[1]))};
        mOpenCLBackend->recordKernel2d(unit.kernel, gws, lws);
    }
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end RasterExecution onResize !\n");
#endif
    return NO_ERROR;
}

bool RasterExecution::CanCombine(const std::vector<Tensor *> &outputs){
    auto des = TensorUtils::getDescribe(outputs[0]);
    auto regions = des->regions;
    if(regions.size() < 2)
        return false;
    auto origin = regions[0].origin;
    const int size0 = regions[0].size[0];
    const int size1 = regions[0].size[1];
    const int size2 = regions[0].size[2];
    const int src_offset = regions[1].src.offset - regions[0].src.offset;
    const int dst_offset = regions[1].dst.offset - regions[0].dst.offset;
    const int src_sride0 = regions[0].src.stride[0];
    const int src_sride1 = regions[0].src.stride[1];
    const int src_sride2 = regions[0].src.stride[2];
    const int dst_sride0 = regions[0].dst.stride[0];
    const int dst_sride1 = regions[0].dst.stride[1];
    const int dst_sride2 = regions[0].dst.stride[2];
    bool res = true;
    for(int i = 1; i < regions.size(); ++i){
        res &= regions[i].origin == origin;
        res &= regions[i].size[0] == size0;
        res &= regions[i].size[1] == size1;
        res &= regions[i].size[2] == size2;
        res &= regions[i].src.stride[0] == src_sride0;
        res &= regions[i].src.stride[1] == src_sride1;
        res &= regions[i].src.stride[2] == src_sride2;
        res &= regions[i].dst.stride[0] == dst_sride0;
        res &= regions[i].dst.stride[1] == dst_sride1;
        res &= regions[i].dst.stride[2] == dst_sride2;
        res &= (regions[i].src.offset - regions[i - 1].src.offset) == src_offset;
        res &= (regions[i].dst.offset - regions[i - 1].dst.offset) == dst_offset;
        if(res == false){
            return res;
        }
    }
    return res;
}

using RasterCreator = TypedCreator<RasterExecution>;
REGISTER_OPENCL_OP_CREATOR(RasterCreator, OpType_Raster, IMAGE);
} // namespace OpenCL
} // namespace MNN
