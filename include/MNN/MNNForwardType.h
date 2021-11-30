//
//  MNNForwardType.h
//  MNN
//
//  Created by MNN on 2019/01/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNForwardType_h
#define MNNForwardType_h
#include <stdint.h>
#include <stddef.h>

typedef enum {
    MNN_FORWARD_CPU = 0,

    /*
     Firtly find the first available backends not equal to CPU
     If no other backends, use cpu
     */
    MNN_FORWARD_AUTO = 4,

    /*Hand write metal*/
    MNN_FORWARD_METAL = 1,

    /*NVIDIA GPU API*/
    MNN_FORWARD_CUDA = 2,

    /*Android / Common Device GPU API*/
    MNN_FORWARD_OPENCL = 3,
    MNN_FORWARD_OPENGL = 6,
    MNN_FORWARD_VULKAN = 7,

    /*Android 8.1's NNAPI, Not Support yet. CoreML Now*/
    MNN_FORWARD_NN = 5,

    /*User can use API from Backend.hpp to add or search Backend*/
    MNN_FORWARD_USER_0 = 8,
    MNN_FORWARD_USER_1 = 9,
    MNN_FORWARD_USER_2 = 10,
    MNN_FORWARD_USER_3 = 11,

    MNN_FORWARD_ALL,

    /* Apply arm extension instruction set to accelerate some Ops, this forward type
       is only used in MNN internal, and will be active automatically when user set forward type
       to be MNN_FORWARD_CPU and extension instruction set is valid on hardware.
    */
    MNN_FORWARD_CPU_EXTENSION

} MNNForwardType;

typedef enum {
    // choose one tuning mode Only
    MNN_GPU_TUNING_NONE    = 1 << 0,/* Forbidden tuning, performance not good */
    MNN_GPU_TUNING_HEAVY  = 1 << 1,/* heavily tuning, usually not suggested */
    MNN_GPU_TUNING_WIDE   = 1 << 2,/* widely tuning, performance good. Default */
    MNN_GPU_TUNING_NORMAL = 1 << 3,/* normal tuning, performance may be ok */
    MNN_GPU_TUNING_FAST   = 1 << 4,/* fast tuning, performance may not good */

    // choose one opencl memory mode Only
    /* User can try OpenCL_MEMORY_BUFFER and OpenCL_MEMORY_IMAGE both,
     then choose the better one according to performance*/
    MNN_GPU_MEMORY_BUFFER = 1 << 6,/* User assign mode */
    MNN_GPU_MEMORY_IMAGE  = 1 << 7,/* User assign mode */
} MNNGpuMode;

#ifdef __cplusplus
namespace MNN {
struct BackendConfig {
    enum MemoryMode { Memory_Normal = 0, Memory_High, Memory_Low };

    MemoryMode memory = Memory_Normal;

    enum PowerMode { Power_Normal = 0, Power_High, Power_Low };

    PowerMode power = Power_Normal;

    enum PrecisionMode { Precision_Normal = 0, Precision_High, Precision_Low };

    PrecisionMode precision = Precision_Normal;

    /** user defined context */
    union {
        void* sharedContext = nullptr;
        size_t flags; // Valid for CPU Backend
    };
};

    /** acquire runtime status by Runtime::getCurrentStatus with following keys,
    */
    enum RuntimeStatus {
        /**
         * get status whether this runtime support 16-bits float point arithmetic
         */
        STATUS_SUPPORT_FP16,
        /**
         * get status whether this runtime support dot-product arithmetic
         */
        STATUS_SUPPORT_DOT_PRODUCT,
        /**
         * emum total number
         */
        STATUS_COUNT
    };


}; // namespace MNN
#endif
#endif /* MNNForwardType_h */
