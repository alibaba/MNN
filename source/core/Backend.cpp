//
//  Backend.cpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Backend.hpp"
#include <stdio.h>
#include <mutex>
#include "MNN_generated.h"
#include "backend/cpu/CPUTensorConvert.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "shape/SizeComputer.hpp"
#ifdef MNN_INTERNAL_ENABLED
#include "internal/logging/Log.hpp"
#endif

namespace MNN {

static std::map<MNNForwardType, std::pair<const RuntimeCreator*, bool>>& GetExtraCreator() {
    static std::once_flag gInitFlag;
    static std::map<MNNForwardType, std::pair<const RuntimeCreator*, bool>>* gExtraCreator;
    std::call_once(gInitFlag,
                   [&]() { gExtraCreator = new std::map<MNNForwardType, std::pair<const RuntimeCreator*, bool>>; });
    return *gExtraCreator;
}

extern void registerCPURuntimeCreator();

#if MNN_METAL_ENABLED
extern void registerMetalRuntimeCreator();
#endif
#if MNN_OPENCL_ENABLED
namespace OpenCL {
extern void registerOpenCLRuntimeCreator();
}
#endif
#if MNN_COREML_ENABLED
extern void registerCoreMLRuntimeCreator();
#endif
#if MNN_NNAPI_ENABLED
extern void registerNNAPIRuntimeCreator();
#endif
#if MNN_QNN_ENABLED
extern void registerQNNRuntimeCreator();
#endif
#ifdef MNN_NEUROPILOT
extern void registerNeuroPilot();
#endif
static std::once_flag s_flag;
void registerBackend() {
    std::call_once(s_flag, [&]() {
#ifdef MNN_INTERNAL_ENABLED
        LogInit();
#endif
        registerCPURuntimeCreator();
#ifndef MNN_SKIPBUILD_GEOMETRY
        SizeComputerSuite::init();
        GeometryComputer::init();
#endif
#if MNN_COREML_ENABLED
        registerCoreMLRuntimeCreator();
#endif
#ifdef MNN_NNAPI_ENABLED
        registerNNAPIRuntimeCreator();
#endif
#if MNN_QNN_ENABLED
    registerQNNRuntimeCreator();
#endif
#if MNN_OPENCL_ENABLED
        OpenCL::registerOpenCLRuntimeCreator();
#endif
#if MNN_METAL_ENABLED
        registerMetalRuntimeCreator();
#endif
#ifdef MNN_NEUROPILOT
        registerNeuroPilot();
#endif
    });
}

const RuntimeCreator* MNNGetExtraRuntimeCreator(MNNForwardType type) {
    registerBackend();

    auto& gExtraCreator = GetExtraCreator();

    // Optional strict mode: disallow creating CPU runtime/creator when running OpenCL-only.
    // This is used to ensure actual computation doesn't silently fall back to CPU.
    // Allowed host-side tensor copies may still happen outside runtime creation.
    static int sStrictNoCpu = -1;
    if (sStrictNoCpu < 0) {
        const char* v = ::getenv("MNN_STRICT_NO_CPU_RUNTIME");
        sStrictNoCpu = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    if (sStrictNoCpu == 1 && type == MNN_FORWARD_CPU) {
        MNN_PRINT("[MNN][STRICT] CPU runtime creation is disabled (MNN_STRICT_NO_CPU_RUNTIME=1).\n");
        return nullptr;
    }

    auto iter           = gExtraCreator.find(type);
    if (iter == gExtraCreator.end()) {
        MNN_PRINT("[MNN] RuntimeCreator not found for type=%d\n", (int)type);
        return nullptr;
    }
    // needCheck == false
    if (!iter->second.second) {
        MNN_PRINT("[MNN] RuntimeCreator found for type=%d (needCheck=0)\n", (int)type);
        return iter->second.first;
    }
    Backend::Info info;
    info.type = type;
    std::shared_ptr<Runtime> bn(iter->second.first->onCreate(info));
    if (nullptr != bn.get()) {
        MNN_PRINT("[MNN] RuntimeCreator validated for type=%d (onCreate ok)\n", (int)type);
        return iter->second.first;
    }
    MNN_PRINT("[MNN] RuntimeCreator present but validation failed for type=%d (onCreate returned null)\n", (int)type);
    return nullptr;
}

bool MNNInsertExtraRuntimeCreator(MNNForwardType type, const RuntimeCreator* creator, bool needCheck) {
    auto& gExtraCreator = GetExtraCreator();
    if (gExtraCreator.find(type) != gExtraCreator.end()) {
        MNN_ASSERT(false && "duplicate type");
        return false;
    }
    gExtraCreator.insert(std::make_pair(type, std::make_pair(creator, needCheck)));
    return true;
}

bool MNNCPUCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) {
    auto& srcBuffer = srcTensor->buffer();
    auto& dstBuffer = dstTensor->buffer();

    MNN_ASSERT(srcBuffer.dimensions == dstBuffer.dimensions);
    MNN_ASSERT(srcBuffer.type == dstBuffer.type);
    if (nullptr == srcBuffer.host || nullptr == dstBuffer.host) {
        return false;
    }
    auto code = CPUTensorConverter::convert(srcTensor, dstTensor);
    if (NO_ERROR != code) {
        MNN_ERROR("Error in CPUBackend::onCopyBuffer\n");
    }
    return true;
}

bool Backend::onAcquireBuffer(const Tensor* tensor, StorageType storageType) {
    auto mem = this->onAcquire(tensor, storageType);
    if (nullptr == mem) {
        return false;
    }
    if (mem == TensorUtils::getDescribeOrigin(tensor)->mem.get()) {
        return true;
    }
    TensorUtils::getDescribeOrigin(tensor)->mem = mem;
    if (nullptr == TensorUtils::getDescribeOrigin(tensor)->getBackend()) {
        TensorUtils::getDescribeOrigin(tensor)->setBackend(this);
    }
    return true;
}
bool Backend::onReleaseBuffer(const Tensor* tensor, StorageType storageType) {
    TensorUtils::getDescribeOrigin(tensor)->mem = nullptr;
    return true;
}

bool Runtime::hasAsyncWork() const {
    return mFuture.valid();
}
void Runtime::setAsyncWork(std::future<int>&& future) {
    mFuture = std::move(future);
}
void Runtime::waitAsyncWork() {
    if (mFuture.valid()) {
        mFuture.wait();
    }
}

} // namespace MNN
