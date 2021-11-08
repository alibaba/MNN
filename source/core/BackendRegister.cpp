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
namespace MNN {
extern void registerCPURuntimeCreator();

#ifdef MNN_CODEGEN_REGISTER
extern void registerArm82RuntimeCreator();
#if MNN_METAL_ENABLED
extern void registerMetalRuntimeCreator();
#endif
#endif
#if MNN_COREML_ENABLED
extern void registerCoreMLRuntimeCreator();
#endif

static std::once_flag s_flag;
void registerBackend() {
    std::call_once(s_flag, [&]() {
        registerCPURuntimeCreator();
#ifndef MNN_BUILD_MINI
        SizeComputerSuite::init();
        GeometryComputer::init();
#endif
#if MNN_COREML_ENABLED
        registerCoreMLRuntimeCreator();
#endif
#ifdef MNN_CODEGEN_REGISTER
#if MNN_METAL_ENABLED
        registerMetalRuntimeCreator();
#endif
#ifdef MNN_USE_ARMV82
        registerArm82RuntimeCreator();
#endif
#endif
    });
}
} // namespace MNN
