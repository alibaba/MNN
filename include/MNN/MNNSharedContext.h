//
//  MNNSharedContext.h
//  MNN
//
//  Created by MNN on 2018/10/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNSharedContext_h
#define MNNSharedContext_h
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h> /*uint32_t*/

#ifndef VK_DEFINE_HANDLE
#define VK_DEFINE_HANDLE(object) typedef struct object##_T* object;
VK_DEFINE_HANDLE(VkInstance)
VK_DEFINE_HANDLE(VkPhysicalDevice)
VK_DEFINE_HANDLE(VkDevice)
VK_DEFINE_HANDLE(VkQueue)
#endif
struct MNNVulkanContext {
    VkInstance pInstance;
    VkPhysicalDevice pPhysicalDevice;
    VkDevice pDevice;
    VkQueue pQueue;
    uint32_t iQueueFamilyIndex;
};
#ifdef __cplusplus
}
#endif

#endif /* MNNSharedContext_h */
