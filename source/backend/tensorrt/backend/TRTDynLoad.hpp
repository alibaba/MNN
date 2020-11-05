//
//  TRTDynLoad.hpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TRTDynLoad_H
#define MNN_TRTDynLoad_H

#include <NvInfer.h>
#include <dlfcn.h>
#include <core/TensorUtils.hpp>
#include <mutex>
#include <string>
#include "TRTSoFinder.hpp"
#include "TRTType.hpp"

namespace MNN {

extern std::once_flag TRTDsoFlag;
extern void* TRTDsoHandle;

#define DECLTYPE(__name, ...) decltype(__name(__VA_ARGS__))

#define DECLARE_DYNLOAD_TRT_WRAP(__name)                                 \
    struct MNNDL__##__name {                                             \
        template <typename... Args>                                      \
        auto operator()(Args... args) -> DECLTYPE(__name, args...) {     \
            using tensorrt_func = decltype(&::__name);                   \
            std::call_once(TRTDsoFlag, []() {                            \
                TRTDsoHandle = GetTRTDsoHandle();                        \
                MNN_ASSERT(TRTDsoHandle != nullptr);                     \
            });                                                          \
            static void* p_##__name = dlsym(TRTDsoHandle, #__name);      \
            MNN_ASSERT(p_##__name != nullptr);                           \
            return reinterpret_cast<tensorrt_func>(p_##__name)(args...); \
        }                                                                \
    };                                                                   \
    extern MNNDL__##__name __name

#define TRT_TYPE_DEFINE(__macro)          \
    __macro(createInferBuilder_INTERNAL); \
    __macro(createInferRuntime_INTERNAL);

TRT_TYPE_DEFINE(DECLARE_DYNLOAD_TRT_WRAP)

static nvinfer1::IBuilder* createInferBuilder(nvinfer1::ILogger& logger) {
    return static_cast<nvinfer1::IBuilder*>(createInferBuilder_INTERNAL(&logger, NV_TENSORRT_VERSION));
}
static nvinfer1::IRuntime* createInferRuntime(nvinfer1::ILogger& logger) {
    return static_cast<nvinfer1::IRuntime*>(createInferRuntime_INTERNAL(&logger, NV_TENSORRT_VERSION));
}

} // namespace MNN

#endif // MNN_TRTDynLoad_H
