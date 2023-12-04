// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// This file is generated.
#include "backend/vulkan/vulkan/vulkan_wrapper.h"
#ifndef MNN_USE_LIB_WRAPPER
int InitVulkan(void) {
    return 1;
}
#else
#include <string>
#include <vector>
#include <mutex>
#ifdef WIN32
#include <windows.h>
#include <libloaderapi.h>
#define MNN_DLSYM(lib, func_name) GetProcAddress(reinterpret_cast<HMODULE>(lib), func_name)
#else
#include <dlfcn.h>
#define MNN_DLSYM(lib, func_name) dlsym(lib, func_name)
#endif

int InitVulkanOnce(void) {
    const std::vector<std::string> gVulkan_library_paths = {
#ifdef WIN32
    "vulkan-1.dll",
#endif
    "libvulkan.so",
#if defined(__APPLE__) || defined(__MACOSX)
    "/usr/local/lib/libvulkan.dylib",// For mac install vk driver
#endif
    };
    void* libvulkan = nullptr;
    for (const auto& s : gVulkan_library_paths) {
#ifdef WIN32
        libvulkan = LoadLibrary(s.c_str());
#else
        libvulkan = dlopen(s.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (nullptr != libvulkan) {
            break;
        }
    }
    if (nullptr == libvulkan) {
#ifdef WIN32
        MNN_ERROR("Load vulkan library error\n");
#else
        auto message = dlerror();
        MNN_ERROR("Load vulkan library error: %s\n", message);
#endif
        return 0;
    }
    // Vulkan supported, set function addresses
    vkCreateInstance  = reinterpret_cast<PFN_vkCreateInstance>(MNN_DLSYM(libvulkan, "vkCreateInstance"));
    vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(MNN_DLSYM(libvulkan, "vkDestroyInstance"));
    vkEnumeratePhysicalDevices =
        reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(MNN_DLSYM(libvulkan, "vkEnumeratePhysicalDevices"));
    vkGetPhysicalDeviceFeatures =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceFeatures"));
    vkGetPhysicalDeviceFormatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceFormatProperties"));
    vkGetPhysicalDeviceImageFormatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceImageFormatProperties"));
    vkGetPhysicalDeviceProperties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceProperties"));
    vkGetPhysicalDeviceQueueFamilyProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceQueueFamilyProperties"));
    vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceMemoryProperties"));
    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(MNN_DLSYM(libvulkan, "vkGetInstanceProcAddr"));
    vkGetDeviceProcAddr   = reinterpret_cast<PFN_vkGetDeviceProcAddr>(MNN_DLSYM(libvulkan, "vkGetDeviceProcAddr"));
    vkCreateDevice        = reinterpret_cast<PFN_vkCreateDevice>(MNN_DLSYM(libvulkan, "vkCreateDevice"));
    vkDestroyDevice       = reinterpret_cast<PFN_vkDestroyDevice>(MNN_DLSYM(libvulkan, "vkDestroyDevice"));
    vkEnumerateInstanceExtensionProperties = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        MNN_DLSYM(libvulkan, "vkEnumerateInstanceExtensionProperties"));
    vkEnumerateDeviceExtensionProperties = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        MNN_DLSYM(libvulkan, "vkEnumerateDeviceExtensionProperties"));
    vkEnumerateInstanceLayerProperties = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
        MNN_DLSYM(libvulkan, "vkEnumerateInstanceLayerProperties"));
    vkEnumerateDeviceLayerProperties =
        reinterpret_cast<PFN_vkEnumerateDeviceLayerProperties>(MNN_DLSYM(libvulkan, "vkEnumerateDeviceLayerProperties"));
    vkGetDeviceQueue = reinterpret_cast<PFN_vkGetDeviceQueue>(MNN_DLSYM(libvulkan, "vkGetDeviceQueue"));
    vkQueueSubmit    = reinterpret_cast<PFN_vkQueueSubmit>(MNN_DLSYM(libvulkan, "vkQueueSubmit"));
    vkQueueWaitIdle  = reinterpret_cast<PFN_vkQueueWaitIdle>(MNN_DLSYM(libvulkan, "vkQueueWaitIdle"));
    vkDeviceWaitIdle = reinterpret_cast<PFN_vkDeviceWaitIdle>(MNN_DLSYM(libvulkan, "vkDeviceWaitIdle"));
    vkAllocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(MNN_DLSYM(libvulkan, "vkAllocateMemory"));
    vkFreeMemory     = reinterpret_cast<PFN_vkFreeMemory>(MNN_DLSYM(libvulkan, "vkFreeMemory"));
    vkMapMemory      = reinterpret_cast<PFN_vkMapMemory>(MNN_DLSYM(libvulkan, "vkMapMemory"));
    vkUnmapMemory    = reinterpret_cast<PFN_vkUnmapMemory>(MNN_DLSYM(libvulkan, "vkUnmapMemory"));
    vkFlushMappedMemoryRanges =
        reinterpret_cast<PFN_vkFlushMappedMemoryRanges>(MNN_DLSYM(libvulkan, "vkFlushMappedMemoryRanges"));
    vkInvalidateMappedMemoryRanges =
        reinterpret_cast<PFN_vkInvalidateMappedMemoryRanges>(MNN_DLSYM(libvulkan, "vkInvalidateMappedMemoryRanges"));
    vkGetDeviceMemoryCommitment =
        reinterpret_cast<PFN_vkGetDeviceMemoryCommitment>(MNN_DLSYM(libvulkan, "vkGetDeviceMemoryCommitment"));
    vkBindBufferMemory = reinterpret_cast<PFN_vkBindBufferMemory>(MNN_DLSYM(libvulkan, "vkBindBufferMemory"));
    vkBindImageMemory  = reinterpret_cast<PFN_vkBindImageMemory>(MNN_DLSYM(libvulkan, "vkBindImageMemory"));
    vkGetBufferMemoryRequirements =
        reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(MNN_DLSYM(libvulkan, "vkGetBufferMemoryRequirements"));
    vkGetImageMemoryRequirements =
        reinterpret_cast<PFN_vkGetImageMemoryRequirements>(MNN_DLSYM(libvulkan, "vkGetImageMemoryRequirements"));
    vkGetImageSparseMemoryRequirements = reinterpret_cast<PFN_vkGetImageSparseMemoryRequirements>(
        MNN_DLSYM(libvulkan, "vkGetImageSparseMemoryRequirements"));
    vkGetPhysicalDeviceSparseImageFormatProperties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSparseImageFormatProperties>(
            MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceSparseImageFormatProperties"));
    vkQueueBindSparse     = reinterpret_cast<PFN_vkQueueBindSparse>(MNN_DLSYM(libvulkan, "vkQueueBindSparse"));
    vkCreateFence         = reinterpret_cast<PFN_vkCreateFence>(MNN_DLSYM(libvulkan, "vkCreateFence"));
    vkDestroyFence        = reinterpret_cast<PFN_vkDestroyFence>(MNN_DLSYM(libvulkan, "vkDestroyFence"));
    vkResetFences         = reinterpret_cast<PFN_vkResetFences>(MNN_DLSYM(libvulkan, "vkResetFences"));
    vkGetFenceStatus      = reinterpret_cast<PFN_vkGetFenceStatus>(MNN_DLSYM(libvulkan, "vkGetFenceStatus"));
    vkWaitForFences       = reinterpret_cast<PFN_vkWaitForFences>(MNN_DLSYM(libvulkan, "vkWaitForFences"));
    vkCreateSemaphore     = reinterpret_cast<PFN_vkCreateSemaphore>(MNN_DLSYM(libvulkan, "vkCreateSemaphore"));
    vkDestroySemaphore    = reinterpret_cast<PFN_vkDestroySemaphore>(MNN_DLSYM(libvulkan, "vkDestroySemaphore"));
    vkCreateEvent         = reinterpret_cast<PFN_vkCreateEvent>(MNN_DLSYM(libvulkan, "vkCreateEvent"));
    vkDestroyEvent        = reinterpret_cast<PFN_vkDestroyEvent>(MNN_DLSYM(libvulkan, "vkDestroyEvent"));
    vkGetEventStatus      = reinterpret_cast<PFN_vkGetEventStatus>(MNN_DLSYM(libvulkan, "vkGetEventStatus"));
    vkSetEvent            = reinterpret_cast<PFN_vkSetEvent>(MNN_DLSYM(libvulkan, "vkSetEvent"));
    vkResetEvent          = reinterpret_cast<PFN_vkResetEvent>(MNN_DLSYM(libvulkan, "vkResetEvent"));
    vkCreateQueryPool     = reinterpret_cast<PFN_vkCreateQueryPool>(MNN_DLSYM(libvulkan, "vkCreateQueryPool"));
    vkDestroyQueryPool    = reinterpret_cast<PFN_vkDestroyQueryPool>(MNN_DLSYM(libvulkan, "vkDestroyQueryPool"));
    vkGetQueryPoolResults = reinterpret_cast<PFN_vkGetQueryPoolResults>(MNN_DLSYM(libvulkan, "vkGetQueryPoolResults"));
    vkCreateBuffer        = reinterpret_cast<PFN_vkCreateBuffer>(MNN_DLSYM(libvulkan, "vkCreateBuffer"));
    vkDestroyBuffer       = reinterpret_cast<PFN_vkDestroyBuffer>(MNN_DLSYM(libvulkan, "vkDestroyBuffer"));
    vkCreateBufferView    = reinterpret_cast<PFN_vkCreateBufferView>(MNN_DLSYM(libvulkan, "vkCreateBufferView"));
    vkDestroyBufferView   = reinterpret_cast<PFN_vkDestroyBufferView>(MNN_DLSYM(libvulkan, "vkDestroyBufferView"));
    vkCreateImage         = reinterpret_cast<PFN_vkCreateImage>(MNN_DLSYM(libvulkan, "vkCreateImage"));
    vkDestroyImage        = reinterpret_cast<PFN_vkDestroyImage>(MNN_DLSYM(libvulkan, "vkDestroyImage"));
    vkGetImageSubresourceLayout =
        reinterpret_cast<PFN_vkGetImageSubresourceLayout>(MNN_DLSYM(libvulkan, "vkGetImageSubresourceLayout"));
    vkCreateImageView      = reinterpret_cast<PFN_vkCreateImageView>(MNN_DLSYM(libvulkan, "vkCreateImageView"));
    vkDestroyImageView     = reinterpret_cast<PFN_vkDestroyImageView>(MNN_DLSYM(libvulkan, "vkDestroyImageView"));
    vkCreateShaderModule   = reinterpret_cast<PFN_vkCreateShaderModule>(MNN_DLSYM(libvulkan, "vkCreateShaderModule"));
    vkDestroyShaderModule  = reinterpret_cast<PFN_vkDestroyShaderModule>(MNN_DLSYM(libvulkan, "vkDestroyShaderModule"));
    vkCreatePipelineCache  = reinterpret_cast<PFN_vkCreatePipelineCache>(MNN_DLSYM(libvulkan, "vkCreatePipelineCache"));
    vkDestroyPipelineCache = reinterpret_cast<PFN_vkDestroyPipelineCache>(MNN_DLSYM(libvulkan, "vkDestroyPipelineCache"));
    vkGetPipelineCacheData = reinterpret_cast<PFN_vkGetPipelineCacheData>(MNN_DLSYM(libvulkan, "vkGetPipelineCacheData"));
    vkMergePipelineCaches  = reinterpret_cast<PFN_vkMergePipelineCaches>(MNN_DLSYM(libvulkan, "vkMergePipelineCaches"));
    vkCreateGraphicsPipelines =
        reinterpret_cast<PFN_vkCreateGraphicsPipelines>(MNN_DLSYM(libvulkan, "vkCreateGraphicsPipelines"));
    vkCreateComputePipelines =
        reinterpret_cast<PFN_vkCreateComputePipelines>(MNN_DLSYM(libvulkan, "vkCreateComputePipelines"));
    vkDestroyPipeline      = reinterpret_cast<PFN_vkDestroyPipeline>(MNN_DLSYM(libvulkan, "vkDestroyPipeline"));
    vkCreatePipelineLayout = reinterpret_cast<PFN_vkCreatePipelineLayout>(MNN_DLSYM(libvulkan, "vkCreatePipelineLayout"));
    vkDestroyPipelineLayout =
        reinterpret_cast<PFN_vkDestroyPipelineLayout>(MNN_DLSYM(libvulkan, "vkDestroyPipelineLayout"));
    vkCreateSampler  = reinterpret_cast<PFN_vkCreateSampler>(MNN_DLSYM(libvulkan, "vkCreateSampler"));
    vkDestroySampler = reinterpret_cast<PFN_vkDestroySampler>(MNN_DLSYM(libvulkan, "vkDestroySampler"));
    vkCreateDescriptorSetLayout =
        reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(MNN_DLSYM(libvulkan, "vkCreateDescriptorSetLayout"));
    vkDestroyDescriptorSetLayout =
        reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(MNN_DLSYM(libvulkan, "vkDestroyDescriptorSetLayout"));
    vkCreateDescriptorPool = reinterpret_cast<PFN_vkCreateDescriptorPool>(MNN_DLSYM(libvulkan, "vkCreateDescriptorPool"));
    vkDestroyDescriptorPool =
        reinterpret_cast<PFN_vkDestroyDescriptorPool>(MNN_DLSYM(libvulkan, "vkDestroyDescriptorPool"));
    vkResetDescriptorPool = reinterpret_cast<PFN_vkResetDescriptorPool>(MNN_DLSYM(libvulkan, "vkResetDescriptorPool"));
    vkAllocateDescriptorSets =
        reinterpret_cast<PFN_vkAllocateDescriptorSets>(MNN_DLSYM(libvulkan, "vkAllocateDescriptorSets"));
    vkFreeDescriptorSets   = reinterpret_cast<PFN_vkFreeDescriptorSets>(MNN_DLSYM(libvulkan, "vkFreeDescriptorSets"));
    vkUpdateDescriptorSets = reinterpret_cast<PFN_vkUpdateDescriptorSets>(MNN_DLSYM(libvulkan, "vkUpdateDescriptorSets"));
    vkCreateFramebuffer    = reinterpret_cast<PFN_vkCreateFramebuffer>(MNN_DLSYM(libvulkan, "vkCreateFramebuffer"));
    vkDestroyFramebuffer   = reinterpret_cast<PFN_vkDestroyFramebuffer>(MNN_DLSYM(libvulkan, "vkDestroyFramebuffer"));
    vkCreateRenderPass     = reinterpret_cast<PFN_vkCreateRenderPass>(MNN_DLSYM(libvulkan, "vkCreateRenderPass"));
    vkDestroyRenderPass    = reinterpret_cast<PFN_vkDestroyRenderPass>(MNN_DLSYM(libvulkan, "vkDestroyRenderPass"));
    vkGetRenderAreaGranularity =
        reinterpret_cast<PFN_vkGetRenderAreaGranularity>(MNN_DLSYM(libvulkan, "vkGetRenderAreaGranularity"));
    vkCreateCommandPool  = reinterpret_cast<PFN_vkCreateCommandPool>(MNN_DLSYM(libvulkan, "vkCreateCommandPool"));
    vkDestroyCommandPool = reinterpret_cast<PFN_vkDestroyCommandPool>(MNN_DLSYM(libvulkan, "vkDestroyCommandPool"));
    vkResetCommandPool   = reinterpret_cast<PFN_vkResetCommandPool>(MNN_DLSYM(libvulkan, "vkResetCommandPool"));
    vkAllocateCommandBuffers =
        reinterpret_cast<PFN_vkAllocateCommandBuffers>(MNN_DLSYM(libvulkan, "vkAllocateCommandBuffers"));
    vkFreeCommandBuffers   = reinterpret_cast<PFN_vkFreeCommandBuffers>(MNN_DLSYM(libvulkan, "vkFreeCommandBuffers"));
    vkBeginCommandBuffer   = reinterpret_cast<PFN_vkBeginCommandBuffer>(MNN_DLSYM(libvulkan, "vkBeginCommandBuffer"));
    vkEndCommandBuffer     = reinterpret_cast<PFN_vkEndCommandBuffer>(MNN_DLSYM(libvulkan, "vkEndCommandBuffer"));
    vkResetCommandBuffer   = reinterpret_cast<PFN_vkResetCommandBuffer>(MNN_DLSYM(libvulkan, "vkResetCommandBuffer"));
    vkCmdBindPipeline      = reinterpret_cast<PFN_vkCmdBindPipeline>(MNN_DLSYM(libvulkan, "vkCmdBindPipeline"));
    vkCmdSetViewport       = reinterpret_cast<PFN_vkCmdSetViewport>(MNN_DLSYM(libvulkan, "vkCmdSetViewport"));
    vkCmdSetScissor        = reinterpret_cast<PFN_vkCmdSetScissor>(MNN_DLSYM(libvulkan, "vkCmdSetScissor"));
    vkCmdSetLineWidth      = reinterpret_cast<PFN_vkCmdSetLineWidth>(MNN_DLSYM(libvulkan, "vkCmdSetLineWidth"));
    vkCmdSetDepthBias      = reinterpret_cast<PFN_vkCmdSetDepthBias>(MNN_DLSYM(libvulkan, "vkCmdSetDepthBias"));
    vkCmdSetBlendConstants = reinterpret_cast<PFN_vkCmdSetBlendConstants>(MNN_DLSYM(libvulkan, "vkCmdSetBlendConstants"));
    vkCmdSetDepthBounds    = reinterpret_cast<PFN_vkCmdSetDepthBounds>(MNN_DLSYM(libvulkan, "vkCmdSetDepthBounds"));
    vkCmdSetStencilCompareMask =
        reinterpret_cast<PFN_vkCmdSetStencilCompareMask>(MNN_DLSYM(libvulkan, "vkCmdSetStencilCompareMask"));
    vkCmdSetStencilWriteMask =
        reinterpret_cast<PFN_vkCmdSetStencilWriteMask>(MNN_DLSYM(libvulkan, "vkCmdSetStencilWriteMask"));
    vkCmdSetStencilReference =
        reinterpret_cast<PFN_vkCmdSetStencilReference>(MNN_DLSYM(libvulkan, "vkCmdSetStencilReference"));
    vkCmdBindDescriptorSets =
        reinterpret_cast<PFN_vkCmdBindDescriptorSets>(MNN_DLSYM(libvulkan, "vkCmdBindDescriptorSets"));
    vkCmdBindIndexBuffer   = reinterpret_cast<PFN_vkCmdBindIndexBuffer>(MNN_DLSYM(libvulkan, "vkCmdBindIndexBuffer"));
    vkCmdBindVertexBuffers = reinterpret_cast<PFN_vkCmdBindVertexBuffers>(MNN_DLSYM(libvulkan, "vkCmdBindVertexBuffers"));
    vkCmdDraw              = reinterpret_cast<PFN_vkCmdDraw>(MNN_DLSYM(libvulkan, "vkCmdDraw"));
    vkCmdDrawIndexed       = reinterpret_cast<PFN_vkCmdDrawIndexed>(MNN_DLSYM(libvulkan, "vkCmdDrawIndexed"));
    vkCmdDrawIndirect      = reinterpret_cast<PFN_vkCmdDrawIndirect>(MNN_DLSYM(libvulkan, "vkCmdDrawIndirect"));
    vkCmdDrawIndexedIndirect =
        reinterpret_cast<PFN_vkCmdDrawIndexedIndirect>(MNN_DLSYM(libvulkan, "vkCmdDrawIndexedIndirect"));
    vkCmdDispatch          = reinterpret_cast<PFN_vkCmdDispatch>(MNN_DLSYM(libvulkan, "vkCmdDispatch"));
    vkCmdDispatchIndirect  = reinterpret_cast<PFN_vkCmdDispatchIndirect>(MNN_DLSYM(libvulkan, "vkCmdDispatchIndirect"));
    vkCmdCopyBuffer        = reinterpret_cast<PFN_vkCmdCopyBuffer>(MNN_DLSYM(libvulkan, "vkCmdCopyBuffer"));
    vkCmdCopyImage         = reinterpret_cast<PFN_vkCmdCopyImage>(MNN_DLSYM(libvulkan, "vkCmdCopyImage"));
    vkCmdBlitImage         = reinterpret_cast<PFN_vkCmdBlitImage>(MNN_DLSYM(libvulkan, "vkCmdBlitImage"));
    vkCmdCopyBufferToImage = reinterpret_cast<PFN_vkCmdCopyBufferToImage>(MNN_DLSYM(libvulkan, "vkCmdCopyBufferToImage"));
    vkCmdCopyImageToBuffer = reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(MNN_DLSYM(libvulkan, "vkCmdCopyImageToBuffer"));
    vkCmdUpdateBuffer      = reinterpret_cast<PFN_vkCmdUpdateBuffer>(MNN_DLSYM(libvulkan, "vkCmdUpdateBuffer"));
    vkCmdFillBuffer        = reinterpret_cast<PFN_vkCmdFillBuffer>(MNN_DLSYM(libvulkan, "vkCmdFillBuffer"));
    vkCmdClearColorImage   = reinterpret_cast<PFN_vkCmdClearColorImage>(MNN_DLSYM(libvulkan, "vkCmdClearColorImage"));
    vkCmdClearDepthStencilImage =
        reinterpret_cast<PFN_vkCmdClearDepthStencilImage>(MNN_DLSYM(libvulkan, "vkCmdClearDepthStencilImage"));
    vkCmdClearAttachments = reinterpret_cast<PFN_vkCmdClearAttachments>(MNN_DLSYM(libvulkan, "vkCmdClearAttachments"));
    vkCmdResolveImage     = reinterpret_cast<PFN_vkCmdResolveImage>(MNN_DLSYM(libvulkan, "vkCmdResolveImage"));
    vkCmdSetEvent         = reinterpret_cast<PFN_vkCmdSetEvent>(MNN_DLSYM(libvulkan, "vkCmdSetEvent"));
    vkCmdResetEvent       = reinterpret_cast<PFN_vkCmdResetEvent>(MNN_DLSYM(libvulkan, "vkCmdResetEvent"));
    vkCmdWaitEvents       = reinterpret_cast<PFN_vkCmdWaitEvents>(MNN_DLSYM(libvulkan, "vkCmdWaitEvents"));
    vkCmdPipelineBarrier  = reinterpret_cast<PFN_vkCmdPipelineBarrier>(MNN_DLSYM(libvulkan, "vkCmdPipelineBarrier"));
    vkCmdBeginQuery       = reinterpret_cast<PFN_vkCmdBeginQuery>(MNN_DLSYM(libvulkan, "vkCmdBeginQuery"));
    vkCmdEndQuery         = reinterpret_cast<PFN_vkCmdEndQuery>(MNN_DLSYM(libvulkan, "vkCmdEndQuery"));
    vkCmdResetQueryPool   = reinterpret_cast<PFN_vkCmdResetQueryPool>(MNN_DLSYM(libvulkan, "vkCmdResetQueryPool"));
    vkCmdWriteTimestamp   = reinterpret_cast<PFN_vkCmdWriteTimestamp>(MNN_DLSYM(libvulkan, "vkCmdWriteTimestamp"));
    vkCmdCopyQueryPoolResults =
        reinterpret_cast<PFN_vkCmdCopyQueryPoolResults>(MNN_DLSYM(libvulkan, "vkCmdCopyQueryPoolResults"));
    vkCmdPushConstants   = reinterpret_cast<PFN_vkCmdPushConstants>(MNN_DLSYM(libvulkan, "vkCmdPushConstants"));
    vkCmdBeginRenderPass = reinterpret_cast<PFN_vkCmdBeginRenderPass>(MNN_DLSYM(libvulkan, "vkCmdBeginRenderPass"));
    vkCmdNextSubpass     = reinterpret_cast<PFN_vkCmdNextSubpass>(MNN_DLSYM(libvulkan, "vkCmdNextSubpass"));
    vkCmdEndRenderPass   = reinterpret_cast<PFN_vkCmdEndRenderPass>(MNN_DLSYM(libvulkan, "vkCmdEndRenderPass"));
    vkCmdExecuteCommands = reinterpret_cast<PFN_vkCmdExecuteCommands>(MNN_DLSYM(libvulkan, "vkCmdExecuteCommands"));
    vkDestroySurfaceKHR  = reinterpret_cast<PFN_vkDestroySurfaceKHR>(MNN_DLSYM(libvulkan, "vkDestroySurfaceKHR"));
    vkGetPhysicalDeviceSurfaceSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceSurfaceSupportKHR"));
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
    vkGetPhysicalDeviceSurfaceFormatsKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
    vkGetPhysicalDeviceSurfacePresentModesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceSurfacePresentModesKHR"));
    vkCreateSwapchainKHR  = reinterpret_cast<PFN_vkCreateSwapchainKHR>(MNN_DLSYM(libvulkan, "vkCreateSwapchainKHR"));
    vkDestroySwapchainKHR = reinterpret_cast<PFN_vkDestroySwapchainKHR>(MNN_DLSYM(libvulkan, "vkDestroySwapchainKHR"));
    vkGetSwapchainImagesKHR =
        reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(MNN_DLSYM(libvulkan, "vkGetSwapchainImagesKHR"));
    vkAcquireNextImageKHR = reinterpret_cast<PFN_vkAcquireNextImageKHR>(MNN_DLSYM(libvulkan, "vkAcquireNextImageKHR"));
    vkQueuePresentKHR     = reinterpret_cast<PFN_vkQueuePresentKHR>(MNN_DLSYM(libvulkan, "vkQueuePresentKHR"));
    vkGetPhysicalDeviceDisplayPropertiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPropertiesKHR>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceDisplayPropertiesKHR"));
    vkGetPhysicalDeviceDisplayPlanePropertiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceDisplayPlanePropertiesKHR"));
    vkGetDisplayPlaneSupportedDisplaysKHR = reinterpret_cast<PFN_vkGetDisplayPlaneSupportedDisplaysKHR>(
        MNN_DLSYM(libvulkan, "vkGetDisplayPlaneSupportedDisplaysKHR"));
    vkGetDisplayModePropertiesKHR =
        reinterpret_cast<PFN_vkGetDisplayModePropertiesKHR>(MNN_DLSYM(libvulkan, "vkGetDisplayModePropertiesKHR"));
    vkCreateDisplayModeKHR = reinterpret_cast<PFN_vkCreateDisplayModeKHR>(MNN_DLSYM(libvulkan, "vkCreateDisplayModeKHR"));
    vkGetDisplayPlaneCapabilitiesKHR =
        reinterpret_cast<PFN_vkGetDisplayPlaneCapabilitiesKHR>(MNN_DLSYM(libvulkan, "vkGetDisplayPlaneCapabilitiesKHR"));
    vkCreateDisplayPlaneSurfaceKHR =
        reinterpret_cast<PFN_vkCreateDisplayPlaneSurfaceKHR>(MNN_DLSYM(libvulkan, "vkCreateDisplayPlaneSurfaceKHR"));
    vkCreateSharedSwapchainsKHR =
        reinterpret_cast<PFN_vkCreateSharedSwapchainsKHR>(MNN_DLSYM(libvulkan, "vkCreateSharedSwapchainsKHR"));

#ifdef VK_USE_PLATFORM_XLIB_KHR
    vkCreateXlibSurfaceKHR = reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(MNN_DLSYM(libvulkan, "vkCreateXlibSurfaceKHR"));
    vkGetPhysicalDeviceXlibPresentationSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceXlibPresentationSupportKHR"));
#endif

#ifdef VK_USE_PLATFORM_XCB_KHR
    vkCreateXcbSurfaceKHR = reinterpret_cast<PFN_vkCreateXcbSurfaceKHR>(MNN_DLSYM(libvulkan, "vkCreateXcbSurfaceKHR"));
    vkGetPhysicalDeviceXcbPresentationSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceXcbPresentationSupportKHR"));
#endif

#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    vkCreateWaylandSurfaceKHR =
        reinterpret_cast<PFN_vkCreateWaylandSurfaceKHR>(MNN_DLSYM(libvulkan, "vkCreateWaylandSurfaceKHR"));
    vkGetPhysicalDeviceWaylandPresentationSupportKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR>(
            MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceWaylandPresentationSupportKHR"));
#endif

#ifdef VK_USE_PLATFORM_MIR_KHR
    vkCreateMirSurfaceKHR = reinterpret_cast<PFN_vkCreateMirSurfaceKHR>(MNN_DLSYM(libvulkan, "vkCreateMirSurfaceKHR"));
    vkGetPhysicalDeviceMirPresentationSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceMirPresentationSupportKHR>(
        MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceMirPresentationSupportKHR"));
#endif

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    vkCreateAndroidSurfaceKHR =
        reinterpret_cast<PFN_vkCreateAndroidSurfaceKHR>(MNN_DLSYM(libvulkan, "vkCreateAndroidSurfaceKHR"));
#endif

#ifdef VK_USE_PLATFORM_WIN32_KHR
    vkCreateWin32SurfaceKHR =
        reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(MNN_DLSYM(libvulkan, "vkCreateWin32SurfaceKHR"));
    vkGetPhysicalDeviceWin32PresentationSupportKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR>(
            MNN_DLSYM(libvulkan, "vkGetPhysicalDeviceWin32PresentationSupportKHR"));
#endif
#ifdef USE_DEBUG_EXTENTIONS
    vkCreateDebugReportCallbackEXT =
        reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(MNN_DLSYM(libvulkan, "vkCreateDebugReportCallbackEXT"));
    vkDestroyDebugReportCallbackEXT =
        reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(MNN_DLSYM(libvulkan, "vkDestroyDebugReportCallbackEXT"));
    vkDebugReportMessageEXT =
        reinterpret_cast<PFN_vkDebugReportMessageEXT>(MNN_DLSYM(libvulkan, "vkDebugReportMessageEXT"));
#endif
    return 1;
}

int InitVulkan(void) {
    static std::once_flag gFlag;
    static int gSuccess = 0;
    std::call_once(gFlag, [] {
        gSuccess = InitVulkanOnce();
    });
    return gSuccess;
}

// No Vulkan support, do not set function addresses
PFN_vkCreateInstance vkCreateInstance;
PFN_vkDestroyInstance vkDestroyInstance;
PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures;
PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties;
PFN_vkGetPhysicalDeviceImageFormatProperties vkGetPhysicalDeviceImageFormatProperties;
PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;
PFN_vkCreateDevice vkCreateDevice;
PFN_vkDestroyDevice vkDestroyDevice;
PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties;
PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties;
PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties;
PFN_vkEnumerateDeviceLayerProperties vkEnumerateDeviceLayerProperties;
PFN_vkGetDeviceQueue vkGetDeviceQueue;
PFN_vkQueueSubmit vkQueueSubmit;
PFN_vkQueueWaitIdle vkQueueWaitIdle;
PFN_vkDeviceWaitIdle vkDeviceWaitIdle;
PFN_vkAllocateMemory vkAllocateMemory;
PFN_vkFreeMemory vkFreeMemory;
PFN_vkMapMemory vkMapMemory;
PFN_vkUnmapMemory vkUnmapMemory;
PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;
PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges;
PFN_vkGetDeviceMemoryCommitment vkGetDeviceMemoryCommitment;
PFN_vkBindBufferMemory vkBindBufferMemory;
PFN_vkBindImageMemory vkBindImageMemory;
PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
PFN_vkGetImageSparseMemoryRequirements vkGetImageSparseMemoryRequirements;
PFN_vkGetPhysicalDeviceSparseImageFormatProperties vkGetPhysicalDeviceSparseImageFormatProperties;
PFN_vkQueueBindSparse vkQueueBindSparse;
PFN_vkCreateFence vkCreateFence;
PFN_vkDestroyFence vkDestroyFence;
PFN_vkResetFences vkResetFences;
PFN_vkGetFenceStatus vkGetFenceStatus;
PFN_vkWaitForFences vkWaitForFences;
PFN_vkCreateSemaphore vkCreateSemaphore;
PFN_vkDestroySemaphore vkDestroySemaphore;
PFN_vkCreateEvent vkCreateEvent;
PFN_vkDestroyEvent vkDestroyEvent;
PFN_vkGetEventStatus vkGetEventStatus;
PFN_vkSetEvent vkSetEvent;
PFN_vkResetEvent vkResetEvent;
PFN_vkCreateQueryPool vkCreateQueryPool;
PFN_vkDestroyQueryPool vkDestroyQueryPool;
PFN_vkGetQueryPoolResults vkGetQueryPoolResults;
PFN_vkCreateBuffer vkCreateBuffer;
PFN_vkDestroyBuffer vkDestroyBuffer;
PFN_vkCreateBufferView vkCreateBufferView;
PFN_vkDestroyBufferView vkDestroyBufferView;
PFN_vkCreateImage vkCreateImage;
PFN_vkDestroyImage vkDestroyImage;
PFN_vkGetImageSubresourceLayout vkGetImageSubresourceLayout;
PFN_vkCreateImageView vkCreateImageView;
PFN_vkDestroyImageView vkDestroyImageView;
PFN_vkCreateShaderModule vkCreateShaderModule;
PFN_vkDestroyShaderModule vkDestroyShaderModule;
PFN_vkCreatePipelineCache vkCreatePipelineCache;
PFN_vkDestroyPipelineCache vkDestroyPipelineCache;
PFN_vkGetPipelineCacheData vkGetPipelineCacheData;
PFN_vkMergePipelineCaches vkMergePipelineCaches;
PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelines;
PFN_vkCreateComputePipelines vkCreateComputePipelines;
PFN_vkDestroyPipeline vkDestroyPipeline;
PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;
PFN_vkCreateSampler vkCreateSampler;
PFN_vkDestroySampler vkDestroySampler;
PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout;
PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout;
PFN_vkCreateDescriptorPool vkCreateDescriptorPool;
PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool;
PFN_vkResetDescriptorPool vkResetDescriptorPool;
PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets;
PFN_vkFreeDescriptorSets vkFreeDescriptorSets;
PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets;
PFN_vkCreateFramebuffer vkCreateFramebuffer;
PFN_vkDestroyFramebuffer vkDestroyFramebuffer;
PFN_vkCreateRenderPass vkCreateRenderPass;
PFN_vkDestroyRenderPass vkDestroyRenderPass;
PFN_vkGetRenderAreaGranularity vkGetRenderAreaGranularity;
PFN_vkCreateCommandPool vkCreateCommandPool;
PFN_vkDestroyCommandPool vkDestroyCommandPool;
PFN_vkResetCommandPool vkResetCommandPool;
PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
PFN_vkFreeCommandBuffers vkFreeCommandBuffers;
PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
PFN_vkEndCommandBuffer vkEndCommandBuffer;
PFN_vkResetCommandBuffer vkResetCommandBuffer;
PFN_vkCmdBindPipeline vkCmdBindPipeline;
PFN_vkCmdSetViewport vkCmdSetViewport;
PFN_vkCmdSetScissor vkCmdSetScissor;
PFN_vkCmdSetLineWidth vkCmdSetLineWidth;
PFN_vkCmdSetDepthBias vkCmdSetDepthBias;
PFN_vkCmdSetBlendConstants vkCmdSetBlendConstants;
PFN_vkCmdSetDepthBounds vkCmdSetDepthBounds;
PFN_vkCmdSetStencilCompareMask vkCmdSetStencilCompareMask;
PFN_vkCmdSetStencilWriteMask vkCmdSetStencilWriteMask;
PFN_vkCmdSetStencilReference vkCmdSetStencilReference;
PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
PFN_vkCmdBindIndexBuffer vkCmdBindIndexBuffer;
PFN_vkCmdBindVertexBuffers vkCmdBindVertexBuffers;
PFN_vkCmdDraw vkCmdDraw;
PFN_vkCmdDrawIndexed vkCmdDrawIndexed;
PFN_vkCmdDrawIndirect vkCmdDrawIndirect;
PFN_vkCmdDrawIndexedIndirect vkCmdDrawIndexedIndirect;
PFN_vkCmdDispatch vkCmdDispatch;
PFN_vkCmdDispatchIndirect vkCmdDispatchIndirect;
PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
PFN_vkCmdCopyImage vkCmdCopyImage;
PFN_vkCmdBlitImage vkCmdBlitImage;
PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImage;
PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer;
PFN_vkCmdUpdateBuffer vkCmdUpdateBuffer;
PFN_vkCmdFillBuffer vkCmdFillBuffer;
PFN_vkCmdClearColorImage vkCmdClearColorImage;
PFN_vkCmdClearDepthStencilImage vkCmdClearDepthStencilImage;
PFN_vkCmdClearAttachments vkCmdClearAttachments;
PFN_vkCmdResolveImage vkCmdResolveImage;
PFN_vkCmdSetEvent vkCmdSetEvent;
PFN_vkCmdResetEvent vkCmdResetEvent;
PFN_vkCmdWaitEvents vkCmdWaitEvents;
PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
PFN_vkCmdBeginQuery vkCmdBeginQuery;
PFN_vkCmdEndQuery vkCmdEndQuery;
PFN_vkCmdResetQueryPool vkCmdResetQueryPool;
PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp;
PFN_vkCmdCopyQueryPoolResults vkCmdCopyQueryPoolResults;
PFN_vkCmdPushConstants vkCmdPushConstants;
PFN_vkCmdBeginRenderPass vkCmdBeginRenderPass;
PFN_vkCmdNextSubpass vkCmdNextSubpass;
PFN_vkCmdEndRenderPass vkCmdEndRenderPass;
PFN_vkCmdExecuteCommands vkCmdExecuteCommands;
PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR;
PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR;
PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR;
PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR;
PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR;
PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR;
PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR;
PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR;
PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR;
PFN_vkQueuePresentKHR vkQueuePresentKHR;
PFN_vkGetPhysicalDeviceDisplayPropertiesKHR vkGetPhysicalDeviceDisplayPropertiesKHR;
PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR vkGetPhysicalDeviceDisplayPlanePropertiesKHR;
PFN_vkGetDisplayPlaneSupportedDisplaysKHR vkGetDisplayPlaneSupportedDisplaysKHR;
PFN_vkGetDisplayModePropertiesKHR vkGetDisplayModePropertiesKHR;
PFN_vkCreateDisplayModeKHR vkCreateDisplayModeKHR;
PFN_vkGetDisplayPlaneCapabilitiesKHR vkGetDisplayPlaneCapabilitiesKHR;
PFN_vkCreateDisplayPlaneSurfaceKHR vkCreateDisplayPlaneSurfaceKHR;
PFN_vkCreateSharedSwapchainsKHR vkCreateSharedSwapchainsKHR;

#ifdef VK_USE_PLATFORM_XLIB_KHR
PFN_vkCreateXlibSurfaceKHR vkCreateXlibSurfaceKHR;
PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR vkGetPhysicalDeviceXlibPresentationSupportKHR;
#endif

#ifdef VK_USE_PLATFORM_XCB_KHR
PFN_vkCreateXcbSurfaceKHR vkCreateXcbSurfaceKHR;
PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR vkGetPhysicalDeviceXcbPresentationSupportKHR;
#endif

#ifdef VK_USE_PLATFORM_WAYLAND_KHR
PFN_vkCreateWaylandSurfaceKHR vkCreateWaylandSurfaceKHR;
PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR vkGetPhysicalDeviceWaylandPresentationSupportKHR;
#endif

#ifdef VK_USE_PLATFORM_MIR_KHR
PFN_vkCreateMirSurfaceKHR vkCreateMirSurfaceKHR;
PFN_vkGetPhysicalDeviceMirPresentationSupportKHR vkGetPhysicalDeviceMirPresentationSupportKHR;
#endif

#ifdef VK_USE_PLATFORM_ANDROID_KHR
PFN_vkCreateAndroidSurfaceKHR vkCreateAndroidSurfaceKHR;
#endif

#ifdef VK_USE_PLATFORM_WIN32_KHR
PFN_vkCreateWin32SurfaceKHR vkCreateWin32SurfaceKHR;
PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR vkGetPhysicalDeviceWin32PresentationSupportKHR;
#endif
PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT;
PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT;
PFN_vkDebugReportMessageEXT vkDebugReportMessageEXT;
#endif
