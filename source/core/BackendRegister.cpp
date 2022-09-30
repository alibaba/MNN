//
//  BackendRegister.cpp
//  MNN
//
//  Created by MNN on 2019/05/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <mutex>
#include "geometry/GeometryComputer.hpp"
#include "shape/SizeComputer.hpp"
#include "Macro.h"
#ifdef MNN_INTERNAL_ENABLED
#include "internal/logging/Log.hpp"
#endif
namespace MNN {
extern void registerCPURuntimeCreator();

#if MNN_METAL_ENABLED
extern void registerMetalRuntimeCreator();
#endif
#if MNN_COREML_ENABLED
extern void registerCoreMLRuntimeCreator();
#endif
#if MNN_NNAPI_ENABLED
extern void registerNNAPIRuntimeCreator();
#endif

static std::once_flag s_flag;
void registerBackend() {
    std::call_once(s_flag, [&]() {
#ifdef MNN_INTERNAL_ENABLED
        LogInit();
#endif
        registerCPURuntimeCreator();
#ifndef MNN_BUILD_MINI
        SizeComputerSuite::init();
        GeometryComputer::init();
#endif
#if MNN_COREML_ENABLED
        registerCoreMLRuntimeCreator();
#endif
#ifdef MNN_NNAPI_ENABLED
        registerNNAPIRuntimeCreator();
#endif
#if MNN_METAL_ENABLED
        registerMetalRuntimeCreator();
#endif
    });
}
} // namespace MNN
