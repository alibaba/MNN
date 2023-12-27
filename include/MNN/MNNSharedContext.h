//
//  MNNSharedContext.h
//  MNN
//
//  Created by MNN on 2018/10/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNSharedContext_h
#define MNNSharedContext_h
#include "MNNDefine.h"
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h> /*uint32_t*/
#ifdef MNN_VULKAN

struct MNNVulkanContext {
    VkInstance pInstance;
    VkPhysicalDevice pPhysicalDevice;
    VkDevice pDevice;
    VkQueue pQueue;
    uint32_t iQueueFamilyIndex;
};

struct MNNVulkanTensorContent {
    VkBuffer buffer;
    VkDeviceSize size;
    VkDeviceSize offset;

    halide_type_t realType;
    int32_t mask; // For future usage
};

#endif

#ifdef MNN_METAL
struct MNNMetalSharedContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

struct MNNMetalTensorContent {
    id<MTLBuffer> buffer;
    int32_t offset;
    id<MTLTexture> texture;
    
    halide_type_t type;
    int32_t mask;
    int32_t forFuture[8];
};

MNN_PUBLIC int MNNMetalGetTensorContent(MNNMetalTensorContent* content, void* tensor);
#endif

#ifdef MNN_USER_SET_DEVICE

struct MNNDeviceContext {
    // When one gpu card has multi devices, choose which device. set deviceId
    uint32_t deviceId = 0;
    // When has multi gpu cards, choose which card. set platformId
    uint32_t platformId = 0;
    // User set number of gpu cards
    uint32_t platformSize = 0;
};

#endif


#ifdef __cplusplus
}
#endif

#endif /* MNNSharedContext_h */
