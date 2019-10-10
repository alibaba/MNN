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

    /*Use IOS's MPS instead of hand-write metal, Not Support yet*/
    MNN_FORWARD_MPS = 2,

    /*Android / Common Device GPU API*/
    MNN_FORWARD_OPENCL = 3,
    MNN_FORWARD_OPENGL = 6,
    MNN_FORWARD_VULKAN = 7,

    /*Android 8.1's NNAPI, Not Support yet*/
    MNN_FORWARD_NN = 5,

    /*User can use API from Backend.hpp to add or search Backend*/
    MNN_FORWARD_USER_0 = 8,
    MNN_FORWARD_USER_1 = 9,
    MNN_FORWARD_USER_2 = 10,
    MNN_FORWARD_USER_3 = 11,

    MNN_FORWARD_ALL
} MNNForwardType;
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
}; // namespace MNN
#endif
#endif /* MNNForwardType_h */
