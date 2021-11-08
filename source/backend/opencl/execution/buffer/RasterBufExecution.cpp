//
//  RasterBufExecution.cpp
//  MNN
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/RasterBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

RasterBufExecution::RasterBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend) {
    mOpenCLBackend = (OpenCLBackend *)backend;
    mOp = op;
    //nothing to do
}

ErrorCode RasterBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("start RasterBufExecution onResize !\n");
#endif
    mTempInput.clear();
    mTempOutput = nullptr;
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    auto des = TensorUtils::getDescribe(input);
    auto outputDes = TensorUtils::getDescribe(output);
    mNeedZero = !TensorUtils::regionIsFull(input);
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
            int region[] = {outputShape[0], ROUND_UP(outputShape[3], 4), outputShape[1], outputShape[2]};//nc4hw
            Unit &unit          = mUnits[kernel_idx++];
            unit.kernel         = runtime->buildKernel("raster", "buffer_set_zero", {});
            unit.localWorkSize  = {8, 8};
            unit.globalWorkSize = {(uint32_t)UP_DIV((region[2] * region[3]), 8)*8,
                                   (uint32_t)UP_DIV((region[0] * region[1]), 8)*8};

            int global_dim0 = region[2] * region[3];
            int global_dim1 = region[0] * region[1];
            
            uint32_t idx   = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel.setArg(idx++, global_dim0);
            ret |= unit.kernel.setArg(idx++, global_dim1);
            ret |= unit.kernel.setArg(idx++, openCLBuffer(output));
            if(ret != CL_SUCCESS)
            {
                MNN_PRINT("setArg err %d\n", (int)ret);
            }
        }
        
        // nc4hw4 buffer raster
        for (auto& slice : des->regions)
        {
            Tensor::InsideDescribe::Region C4Region;
            OpCommonUtils::turnToPackRegion(slice, C4Region, output, 4);

            Unit &unit          = mUnits[kernel_idx++];
            unit.kernel         = runtime->buildKernel("raster_buf", "raster_nc4hw4_buffer", {});

            const std::vector<uint32_t> gws =  {(uint32_t)C4Region.size[2],
                                                    (uint32_t)C4Region.size[1],
                                                    (uint32_t)C4Region.size[0]};
            uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

            auto outputShape    = tensorShapeFormat(output);
            auto sliceShape    = tensorShapeFormat(slice.origin);

            uint32_t idx   = 0;
            cl_int ret = CL_SUCCESS;
            ret |= unit.kernel.setArg(idx++, gws[0]);
            ret |= unit.kernel.setArg(idx++, gws[1]);
            ret |= unit.kernel.setArg(idx++, gws[2]);
            ret |= unit.kernel.setArg(idx++, openCLBuffer(slice.origin));
            ret |= unit.kernel.setArg(idx++, C4Region.src.offset);
            ret |= unit.kernel.setArg(idx++, C4Region.src.stride[0]);
            ret |= unit.kernel.setArg(idx++, C4Region.src.stride[1]);
            ret |= unit.kernel.setArg(idx++, C4Region.src.stride[2]);
            ret |= unit.kernel.setArg(idx++, sliceShape[1]);
            ret |= unit.kernel.setArg(idx++, sliceShape[2]);
            ret |= unit.kernel.setArg(idx++, sliceShape[3]);
            ret |= unit.kernel.setArg(idx++, openCLBuffer(output));
            ret |= unit.kernel.setArg(idx++, C4Region.dst.offset);
            ret |= unit.kernel.setArg(idx++, C4Region.dst.stride[0]);
            ret |= unit.kernel.setArg(idx++, C4Region.dst.stride[1]);
            ret |= unit.kernel.setArg(idx++, C4Region.dst.stride[2]);
            ret |= unit.kernel.setArg(idx++, outputShape[1]);
            ret |= unit.kernel.setArg(idx++, outputShape[2]);
            ret |= unit.kernel.setArg(idx++, outputShape[3]);
            if(ret != CL_SUCCESS)
            {
                MNN_PRINT("setArg err %d\n", (int)ret);
            }
            std::string name = "raster_nc4hw4_buffer";
            const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
            
            unit.localWorkSize = {lws[0], lws[1], lws[2]};
            
            unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
                                   ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
                                   ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]))};
        }
        if(mNeedZero)
        {
            MNN_ASSERT((regionNum+1==kernel_idx));
        }
        else
        {
            MNN_ASSERT((regionNum==kernel_idx));
        }
        return NO_ERROR;
    }

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
    mUnits.resize(regionNum + originNum + 1);
    
    int kernel_idx = 0;
    if(mNeedZero)
    {
        mUnits.resize(regionNum + originNum + 2);
        auto outputShape    = tensorShapeFormat(output);
        int region[] = {outputShape[0], ROUND_UP(outputShape[3], 4), outputShape[1], outputShape[2]};//nhwc
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster", "buffer_set_zero", {});
        unit.localWorkSize  = {8, 8};
        unit.globalWorkSize = {(uint32_t)UP_DIV((region[2] * region[3]), 8)*8,
                               (uint32_t)UP_DIV((region[0] * region[1]), 8)*8};
        
        int global_dim0 = region[2] * region[3];
        int global_dim1 = region[0] * region[1];
        
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel.setArg(idx++, global_dim0);
        ret |= unit.kernel.setArg(idx++, global_dim1);
        ret |= unit.kernel.setArg(idx++, *mTempOutput);
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
    }

    // nc4hw4 buffer to buffer
    for(auto& iter : mTempInput)
    {
        Tensor* origin = iter.first;
        std::vector<int> regionShape = tensorShapeFormat(origin);
        int inputWH[]      = {regionShape[2], regionShape[1]};
        int region[]       = {regionShape[0], UP_DIV(regionShape[3], 4), regionShape[1], regionShape[2]};
                
        Unit &unit          = mUnits[kernel_idx++];
        
        if(TensorUtils::getDescribe(origin)->dimensionFormat == MNN_DATA_FORMAT_NHWC)// Image to nhwc buffer
        {
            unit.kernel         = runtime->buildKernel("buffer_convert_buf", "nc4hw4_buffer_to_nhwc_buffer", {});
        }
        else //Image to nchw buffer
        {
            unit.kernel         = runtime->buildKernel("buffer_convert_buf", "nc4hw4_buffer_to_nchw_buffer", {});
        }
        unit.localWorkSize  = {16, 16};
        unit.globalWorkSize = {(uint32_t)UP_DIV(region[3] * region[1], 16) * 16,
                               (uint32_t)UP_DIV(region[2] * region[0], 16) * 16};
        
        int global_dim0 = region[3] * region[1];
        int global_dim1 = region[2] * region[0];

        //MNN_CHECK_CL_SUCCESS
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel.setArg(idx++, global_dim0);
        ret |= unit.kernel.setArg(idx++, global_dim1);
        ret |= unit.kernel.setArg(idx++, *(iter.second));
        ret |= unit.kernel.setArg(idx++, inputWH[1]);
        ret |= unit.kernel.setArg(idx++, inputWH[0]);
        ret |= unit.kernel.setArg(idx++, regionShape[3]);
        ret |= unit.kernel.setArg(idx++, openCLBuffer(origin));
        
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
    }
    
    // buffer raster
    for (auto& slice : des->regions)
    {
        Unit &unit          = mUnits[kernel_idx++];
        unit.kernel         = runtime->buildKernel("raster_buf", "raster_buffer", {});

        const std::vector<uint32_t> gws =  {(uint32_t)slice.size[2],
                                                (uint32_t)slice.size[1],
                                                (uint32_t)slice.size[0]};
        uint32_t mMaxWorkGroupSize      = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel.setArg(idx++, gws[0]);
        ret |= unit.kernel.setArg(idx++, gws[1]);
        ret |= unit.kernel.setArg(idx++, gws[2]);
        ret |= unit.kernel.setArg(idx++, *(mTempInput[slice.origin]));
        ret |= unit.kernel.setArg(idx++, slice.src.offset);
        ret |= unit.kernel.setArg(idx++, slice.src.stride[0]);
        ret |= unit.kernel.setArg(idx++, slice.src.stride[1]);
        ret |= unit.kernel.setArg(idx++, slice.src.stride[2]);
        ret |= unit.kernel.setArg(idx++, *mTempOutput);
        ret |= unit.kernel.setArg(idx++, slice.dst.offset);
        ret |= unit.kernel.setArg(idx++, slice.dst.stride[0]);
        ret |= unit.kernel.setArg(idx++, slice.dst.stride[1]);
        ret |= unit.kernel.setArg(idx++, slice.dst.stride[2]);
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
        
        std::string name = "raster_buffer";
        const std::vector<uint32_t> lws = localWS3DDefault(gws, mMaxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel).first;
        
        unit.localWorkSize = {lws[0], lws[1], lws[2]};
        
        unit.globalWorkSize = {ROUND_UP(gws[0], std::max((uint32_t)1, lws[0])),
                               ROUND_UP(gws[1], std::max((uint32_t)1, lws[1])),
                               ROUND_UP(gws[2], std::max((uint32_t)1, lws[2]))};
    }
    
    //buffer to nc4hw4 buffer
    {
        auto outputShape    = tensorShapeFormat(output);
        int wh[]     = {outputShape[2], outputShape[1]};
        int region[] = {outputShape[0], UP_DIV(outputShape[3], 4), outputShape[1], outputShape[2]};

        Unit &unit          = mUnits[kernel_idx++];
        if(outputDes->dimensionFormat == MNN_DATA_FORMAT_NHWC)//nhwc buffer to Image
        {
            unit.kernel         = runtime->buildKernel("buffer_convert_buf", "nhwc_buffer_to_nc4hw4_buffer", {});
        }
        else //nchw buffer to Image
        {
            unit.kernel         = runtime->buildKernel("buffer_convert_buf", "nchw_buffer_to_nc4hw4_buffer", {});
        }
        unit.localWorkSize  = {16, 16};
        unit.globalWorkSize = {(uint32_t)UP_DIV(region[3] * region[1], 16) * 16,
                               (uint32_t)UP_DIV(region[2] * region[0], 16) * 16};
        
        int global_dim0 = region[3] * region[1];
        int global_dim1 = region[2] * region[0];
        
        uint32_t idx   = 0;
        cl_int ret = CL_SUCCESS;
        ret |= unit.kernel.setArg(idx++, global_dim0);
        ret |= unit.kernel.setArg(idx++, global_dim1);
        ret |= unit.kernel.setArg(idx++, *mTempOutput);
        ret |= unit.kernel.setArg(idx++, wh[1]);
        ret |= unit.kernel.setArg(idx++, wh[0]);
        ret |= unit.kernel.setArg(idx++, outputShape[3]);
        ret |= unit.kernel.setArg(idx++, openCLBuffer(output));
        if(ret != CL_SUCCESS)
        {
            MNN_PRINT("setArg err %d\n", (int)ret);
        }
    }
    
    //kernel num check
    if(mNeedZero)
    {
        MNN_ASSERT((kernel_idx==regionNum + originNum + 2));
    }
    else
    {
        MNN_ASSERT((kernel_idx==regionNum + originNum + 1));
    }
    
#ifdef LOG_VERBOSE
    MNN_PRINT("end RasterBufExecution onResize !\n");
#endif
    return NO_ERROR;
}

class RasterCreator : public OpenCLBackend::Creator {
public:
    virtual ~RasterCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op,
                                Backend *backend) const override {
        return new RasterBufExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<RasterCreator> __RasterBuf_op(OpType_Raster, BUFFER);
} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
