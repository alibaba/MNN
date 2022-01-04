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
    int32_t forFuture[8];
};

MNN_PUBLIC int MNNMetalGetTensorContent(MNNMetalTensorContent* content, void* tensor);
#endif


#ifdef __cplusplus
}
#endif

#endif /* MNNSharedContext_h */
