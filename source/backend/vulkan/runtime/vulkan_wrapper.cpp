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
#include "vulkan_wrapper.h"
#include <dlfcn.h>

int InitVulkan(void) {
    void* libvulkan = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!libvulkan)
        return 0;

    // Vulkan supported, set function addresses
    vkCreateInstance  = reinterpret_cast<PFN_vkCreateInstance>(dlsym(libvulkan, "vkCreateInstance"));
    vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(dlsym(libvulkan, "vkDestroyInstance"));
    vkEnumeratePhysicalDevices =
        reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(dlsym(libvulkan, "vkEnumeratePhysicalDevices"));
    vkGetPhysicalDeviceFeatures =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(dlsym(libvulkan, "vkGetPhysicalDeviceFeatures"));
    vkGetPhysicalDeviceFormatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
        dlsym(libvulkan, "vkGetPhysicalDeviceFormatProperties"));
    vkGetPhysicalDeviceImageFormatProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties>(
        dlsym(libvulkan, "vkGetPhysicalDeviceImageFormatProperties"));
    vkGetPhysicalDeviceProperties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(dlsym(libvulkan, "vkGetPhysicalDeviceProperties"));
    vkGetPhysicalDeviceQueueFamilyProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        dlsym(libvulkan, "vkGetPhysicalDeviceQueueFamilyProperties"));
    vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        dlsym(libvulkan, "vkGetPhysicalDeviceMemoryProperties"));
    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(libvulkan, "vkGetInstanceProcAddr"));
    vkGetDeviceProcAddr   = reinterpret_cast<PFN_vkGetDeviceProcAddr>(dlsym(libvulkan, "vkGetDeviceProcAddr"));
    vkCreateDevice        = reinterpret_cast<PFN_vkCreateDevice>(dlsym(libvulkan, "vkCreateDevice"));
    vkDestroyDevice       = reinterpret_cast<PFN_vkDestroyDevice>(dlsym(libvulkan, "vkDestroyDevice"));
    vkEnumerateInstanceExtensionProperties = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        dlsym(libvulkan, "vkEnumerateInstanceExtensionProperties"));
    vkEnumerateDeviceExtensionProperties = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        dlsym(libvulkan, "vkEnumerateDeviceExtensionProperties"));
    vkEnumerateInstanceLayerProperties = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
        dlsym(libvulkan, "vkEnumerateInstanceLayerProperties"));
    vkEnumerateDeviceLayerProperties =
        reinterpret_cast<PFN_vkEnumerateDeviceLayerProperties>(dlsym(libvulkan, "vkEnumerateDeviceLayerProperties"));
    vkGetDeviceQueue = reinterpret_cast<PFN_vkGetDeviceQueue>(dlsym(libvulkan, "vkGetDeviceQueue"));
    vkQueueSubmit    = reinterpret_cast<PFN_vkQueueSubmit>(dlsym(libvulkan, "vkQueueSubmit"));
    vkQueueWaitIdle  = reinterpret_cast<PFN_vkQueueWaitIdle>(dlsym(libvulkan, "vkQueueWaitIdle"));
    vkDeviceWaitIdle = reinterpret_cast<PFN_vkDeviceWaitIdle>(dlsym(libvulkan, "vkDeviceWaitIdle"));
    vkAllocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(dlsym(libvulkan, "vkAllocateMemory"));
    vkFreeMemory     = reinterpret_cast<PFN_vkFreeMemory>(dlsym(libvulkan, "vkFreeMemory"));
    vkMapMemory      = reinterpret_cast<PFN_vkMapMemory>(dlsym(libvulkan, "vkMapMemory"));
    vkUnmapMemory    = reinterpret_cast<PFN_vkUnmapMemory>(dlsym(libvulkan, "vkUnmapMemory"));
    vkFlushMappedMemoryRanges =
        reinterpret_cast<PFN_vkFlushMappedMemoryRanges>(dlsym(libvulkan, "vkFlushMappedMemoryRanges"));
    vkInvalidateMappedMemoryRanges =
        reinterpret_cast<PFN_vkInvalidateMappedMemoryRanges>(dlsym(libvulkan, "vkInvalidateMappedMemoryRanges"));
    vkGetDeviceMemoryCommitment =
        reinterpret_cast<PFN_vkGetDeviceMemoryCommitment>(dlsym(libvulkan, "vkGetDeviceMemoryCommitment"));
    vkBindBufferMemory = reinterpret_cast<PFN_vkBindBufferMemory>(dlsym(libvulkan, "vkBindBufferMemory"));
    vkBindImageMemory  = reinterpret_cast<PFN_vkBindImageMemory>(dlsym(libvulkan, "vkBindImageMemory"));
    vkGetBufferMemoryRequirements =
        reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(dlsym(libvulkan, "vkGetBufferMemoryRequirements"));
    vkGetImageMemoryRequirements =
        reinterpret_cast<PFN_vkGetImageMemoryRequirements>(dlsym(libvulkan, "vkGetImageMemoryRequirements"));
    vkGetImageSparseMemoryRequirements = reinterpret_cast<PFN_vkGetImageSparseMemoryRequirements>(
        dlsym(libvulkan, "vkGetImageSparseMemoryRequirements"));
    vkGetPhysicalDeviceSparseImageFormatProperties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceSparseImageFormatProperties>(
            dlsym(libvulkan, "vkGetPhysicalDeviceSparseImageFormatProperties"));
    vkQueueBindSparse     = reinterpret_cast<PFN_vkQueueBindSparse>(dlsym(libvulkan, "vkQueueBindSparse"));
    vkCreateFence         = reinterpret_cast<PFN_vkCreateFence>(dlsym(libvulkan, "vkCreateFence"));
    vkDestroyFence        = reinterpret_cast<PFN_vkDestroyFence>(dlsym(libvulkan, "vkDestroyFence"));
    vkResetFences         = reinterpret_cast<PFN_vkResetFences>(dlsym(libvulkan, "vkResetFences"));
    vkGetFenceStatus      = reinterpret_cast<PFN_vkGetFenceStatus>(dlsym(libvulkan, "vkGetFenceStatus"));
    vkWaitForFences       = reinterpret_cast<PFN_vkWaitForFences>(dlsym(libvulkan, "vkWaitForFences"));
    vkCreateSemaphore     = reinterpret_cast<PFN_vkCreateSemaphore>(dlsym(libvulkan, "vkCreateSemaphore"));
    vkDestroySemaphore    = reinterpret_cast<PFN_vkDestroySemaphore>(dlsym(libvulkan, "vkDestroySemaphore"));
    vkCreateEvent         = reinterpret_cast<PFN_vkCreateEvent>(dlsym(libvulkan, "vkCreateEvent"));
    vkDestroyEvent        = reinterpret_cast<PFN_vkDestroyEvent>(dlsym(libvulkan, "vkDestroyEvent"));
    vkGetEventStatus      = reinterpret_cast<PFN_vkGetEventStatus>(dlsym(libvulkan, "vkGetEventStatus"));
    vkSetEvent            = reinterpret_cast<PFN_vkSetEvent>(dlsym(libvulkan, "vkSetEvent"));
    vkResetEvent          = reinterpret_cast<PFN_vkResetEvent>(dlsym(libvulkan, "vkResetEvent"));
    vkCreateQueryPool     = reinterpret_cast<PFN_vkCreateQueryPool>(dlsym(libvulkan, "vkCreateQueryPool"));
    vkDestroyQueryPool    = reinterpret_cast<PFN_vkDestroyQueryPool>(dlsym(libvulkan, "vkDestroyQueryPool"));
    vkGetQueryPoolResults = reinterpret_cast<PFN_vkGetQueryPoolResults>(dlsym(libvulkan, "vkGetQueryPoolResults"));
    vkCreateBuffer        = reinterpret_cast<PFN_vkCreateBuffer>(dlsym(libvulkan, "vkCreateBuffer"));
    vkDestroyBuffer       = reinterpret_cast<PFN_vkDestroyBuffer>(dlsym(libvulkan, "vkDestroyBuffer"));
    vkCreateBufferView    = reinterpret_cast<PFN_vkCreateBufferView>(dlsym(libvulkan, "vkCreateBufferView"));
    vkDestroyBufferView   = reinterpret_cast<PFN_vkDestroyBufferView>(dlsym(libvulkan, "vkDestroyBufferView"));
    vkCreateImage         = reinterpret_cast<PFN_vkCreateImage>(dlsym(libvulkan, "vkCreateImage"));
    vkDestroyImage        = reinterpret_cast<PFN_vkDestroyImage>(dlsym(libvulkan, "vkDestroyImage"));
    vkGetImageSubresourceLayout =
        reinterpret_cast<PFN_vkGetImageSubresourceLayout>(dlsym(libvulkan, "vkGetImageSubresourceLayout"));
    vkCreateImageView      = reinterpret_cast<PFN_vkCreateImageView>(dlsym(libvulkan, "vkCreateImageView"));
    vkDestroyImageView     = reinterpret_cast<PFN_vkDestroyImageView>(dlsym(libvulkan, "vkDestroyImageView"));
    vkCreateShaderModule   = reinterpret_cast<PFN_vkCreateShaderModule>(dlsym(libvulkan, "vkCreateShaderModule"));
    vkDestroyShaderModule  = reinterpret_cast<PFN_vkDestroyShaderModule>(dlsym(libvulkan, "vkDestroyShaderModule"));
    vkCreatePipelineCache  = reinterpret_cast<PFN_vkCreatePipelineCache>(dlsym(libvulkan, "vkCreatePipelineCache"));
    vkDestroyPipelineCache = reinterpret_cast<PFN_vkDestroyPipelineCache>(dlsym(libvulkan, "vkDestroyPipelineCache"));
    vkGetPipelineCacheData = reinterpret_cast<PFN_vkGetPipelineCacheData>(dlsym(libvulkan, "vkGetPipelineCacheData"));
    vkMergePipelineCaches  = reinterpret_cast<PFN_vkMergePipelineCaches>(dlsym(libvulkan, "vkMergePipelineCaches"));
    vkCreateGraphicsPipelines =
        reinterpret_cast<PFN_vkCreateGraphicsPipelines>(dlsym(libvulkan, "vkCreateGraphicsPipelines"));
    vkCreateComputePipelines =
        reinterpret_cast<PFN_vkCreateComputePipelines>(dlsym(libvulkan, "vkCreateComputePipelines"));
    vkDestroyPipeline      = reinterpret_cast<PFN_vkDestroyPipeline>(dlsym(libvulkan, "vkDestroyPipeline"));
    vkCreatePipelineLayout = reinterpret_cast<PFN_vkCreatePipelineLayout>(dlsym(libvulkan, "vkCreatePipelineLayout"));
    vkDestroyPipelineLayout =
        reinterpret_cast<PFN_vkDestroyPipelineLayout>(dlsym(libvulkan, "vkDestroyPipelineLayout"));
    vkCreateSampler  = reinterpret_cast<PFN_vkCreateSampler>(dlsym(libvulkan, "vkCreateSampler"));
    vkDestroySampler = reinterpret_cast<PFN_vkDestroySampler>(dlsym(libvulkan, "vkDestroySampler"));
    vkCreateDescriptorSetLayout =
        reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(dlsym(libvulkan, "vkCreateDescriptorSetLayout"));
    vkDestroyDescriptorSetLayout =
        reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(dlsym(libvulkan, "vkDestroyDescriptorSetLayout"));
    vkCreateDescriptorPool = reinterpret_cast<PFN_vkCreateDescriptorPool>(dlsym(libvulkan, "vkCreateDescriptorPool"));
    vkDestroyDescriptorPool =
        reinterpret_cast<PFN_vkDestroyDescriptorPool>(dlsym(libvulkan, "vkDestroyDescriptorPool"));
    vkResetDescriptorPool = reinterpret_cast<PFN_vkResetDescriptorPool>(dlsym(libvulkan, "vkResetDescriptorPool"));
    vkAllocateDescriptorSets =
        reinterpret_cast<PFN_vkAllocateDescriptorSets>(dlsym(libvulkan, "vkAllocateDescriptorSets"));
    vkFreeDescriptorSets   = reinterpret_cast<PFN_vkFreeDescriptorSets>(dlsym(libvulkan, "vkFreeDescriptorSets"));
    vkUpdateDescriptorSets = reinterpret_cast<PFN_vkUpdateDescriptorSets>(dlsym(libvulkan, "vkUpdateDescriptorSets"));
    vkCreateFramebuffer    = reinterpret_cast<PFN_vkCreateFramebuffer>(dlsym(libvulkan, "vkCreateFramebuffer"));
    vkDestroyFramebuffer   = reinterpret_cast<PFN_vkDestroyFramebuffer>(dlsym(libvulkan, "vkDestroyFramebuffer"));
    vkCreateRenderPass     = reinterpret_cast<PFN_vkCreateRenderPass>(dlsym(libvulkan, "vkCreateRenderPass"));
    vkDestroyRenderPass    = reinterpret_cast<PFN_vkDestroyRenderPass>(dlsym(libvulkan, "vkDestroyRenderPass"));
    vkGetRenderAreaGranularity =
        reinterpret_cast<PFN_vkGetRenderAreaGranularity>(dlsym(libvulkan, "vkGetRenderAreaGranularity"));
    vkCreateCommandPool  = reinterpret_cast<PFN_vkCreateCommandPool>(dlsym(libvulkan, "vkCreateCommandPool"));
    vkDestroyCommandPool = reinterpret_cast<PFN_vkDestroyCommandPool>(dlsym(libvulkan, "vkDestroyCommandPool"));
    vkResetCommandPool   = reinterpret_cast<PFN_vkResetCommandPool>(dlsym(libvulkan, "vkResetCommandPool"));
    vkAllocateCommandBuffers =
        reinterpret_cast<PFN_vkAllocateCommandBuffers>(dlsym(libvulkan, "vkAllocateCommandBuffers"));
    vkFreeCommandBuffers   = reinterpret_cast<PFN_vkFreeCommandBuffers>(dlsym(libvulkan, "vkFreeCommandBuffers"));
    vkBeginCommandBuffer   = reinterpret_cast<PFN_vkBeginCommandBuffer>(dlsym(libvulkan, "vkBeginCommandBuffer"));
    vkEndCommandBuffer     = reinterpret_cast<PFN_vkEndCommandBuffer>(dlsym(libvulkan, "vkEndCommandBuffer"));
    vkResetCommandBuffer   = reinterpret_cast<PFN_vkResetCommandBuffer>(dlsym(libvulkan, "vkResetCommandBuffer"));
    vkCmdBindPipeline      = reinterpret_cast<PFN_vkCmdBindPipeline>(dlsym(libvulkan, "vkCmdBindPipeline"));
    vkCmdSetViewport       = reinterpret_cast<PFN_vkCmdSetViewport>(dlsym(libvulkan, "vkCmdSetViewport"));
    vkCmdSetScissor        = reinterpret_cast<PFN_vkCmdSetScissor>(dlsym(libvulkan, "vkCmdSetScissor"));
    vkCmdSetLineWidth      = reinterpret_cast<PFN_vkCmdSetLineWidth>(dlsym(libvulkan, "vkCmdSetLineWidth"));
    vkCmdSetDepthBias      = reinterpret_cast<PFN_vkCmdSetDepthBias>(dlsym(libvulkan, "vkCmdSetDepthBias"));
    vkCmdSetBlendConstants = reinterpret_cast<PFN_vkCmdSetBlendConstants>(dlsym(libvulkan, "vkCmdSetBlendConstants"));
    vkCmdSetDepthBounds    = reinterpret_cast<PFN_vkCmdSetDepthBounds>(dlsym(libvulkan, "vkCmdSetDepthBounds"));
    vkCmdSetStencilCompareMask =
        reinterpret_cast<PFN_vkCmdSetStencilCompareMask>(dlsym(libvulkan, "vkCmdSetStencilCompareMask"));
    vkCmdSetStencilWriteMask =
        reinterpret_cast<PFN_vkCmdSetStencilWriteMask>(dlsym(libvulkan, "vkCmdSetStencilWriteMask"));
    vkCmdSetStencilReference =
        reinterpret_cast<PFN_vkCmdSetStencilReference>(dlsym(libvulkan, "vkCmdSetStencilReference"));
    vkCmdBindDescriptorSets =
        reinterpret_cast<PFN_vkCmdBindDescriptorSets>(dlsym(libvulkan, "vkCmdBindDescriptorSets"));
    vkCmdBindIndexBuffer   = reinterpret_cast<PFN_vkCmdBindIndexBuffer>(dlsym(libvulkan, "vkCmdBindIndexBuffer"));
    vkCmdBindVertexBuffers = reinterpret_cast<PFN_vkCmdBindVertexBuffers>(dlsym(libvulkan, "vkCmdBindVertexBuffers"));
    vkCmdDraw              = reinterpret_cast<PFN_vkCmdDraw>(dlsym(libvulkan, "vkCmdDraw"));
    vkCmdDrawIndexed       = reinterpret_cast<PFN_vkCmdDrawIndexed>(dlsym(libvulkan, "vkCmdDrawIndexed"));
    vkCmdDrawIndirect      = reinterpret_cast<PFN_vkCmdDrawIndirect>(dlsym(libvulkan, "vkCmdDrawIndirect"));
    vkCmdDrawIndexedIndirect =
        reinterpret_cast<PFN_vkCmdDrawIndexedIndirect>(dlsym(libvulkan, "vkCmdDrawIndexedIndirect"));
    vkCmdDispatch          = reinterpret_cast<PFN_vkCmdDispatch>(dlsym(libvulkan, "vkCmdDispatch"));
    vkCmdDispatchIndirect  = reinterpret_cast<PFN_vkCmdDispatchIndirect>(dlsym(libvulkan, "vkCmdDispatchIndirect"));
    vkCmdCopyBuffer        = reinterpret_cast<PFN_vkCmdCopyBuffer>(dlsym(libvulkan, "vkCmdCopyBuffer"));
    vkCmdCopyImage         = reinterpret_cast<PFN_vkCmdCopyImage>(dlsym(libvulkan, "vkCmdCopyImage"));
    vkCmdBlitImage         = reinterpret_cast<PFN_vkCmdBlitImage>(dlsym(libvulkan, "vkCmdBlitImage"));
    vkCmdCopyBufferToImage = reinterpret_cast<PFN_vkCmdCopyBufferToImage>(dlsym(libvulkan, "vkCmdCopyBufferToImage"));
    vkCmdCopyImageToBuffer = reinterpret_cast<PFN_vkCmdCopyImageToBuffer>(dlsym(libvulkan, "vkCmdCopyImageToBuffer"));
    vkCmdUpdateBuffer      = reinterpret_cast<PFN_vkCmdUpdateBuffer>(dlsym(libvulkan, "vkCmdUpdateBuffer"));
    vkCmdFillBuffer        = reinterpret_cast<PFN_vkCmdFillBuffer>(dlsym(libvulkan, "vkCmdFillBuffer"));
    vkCmdClearColorImage   = reinterpret_cast<PFN_vkCmdClearColorImage>(dlsym(libvulkan, "vkCmdClearColorImage"));
    vkCmdClearDepthStencilImage =
        reinterpret_cast<PFN_vkCmdClearDepthStencilImage>(dlsym(libvulkan, "vkCmdClearDepthStencilImage"));
    vkCmdClearAttachments = reinterpret_cast<PFN_vkCmdClearAttachments>(dlsym(libvulkan, "vkCmdClearAttachments"));
    vkCmdResolveImage     = reinterpret_cast<PFN_vkCmdResolveImage>(dlsym(libvulkan, "vkCmdResolveImage"));
    vkCmdSetEvent         = reinterpret_cast<PFN_vkCmdSetEvent>(dlsym(libvulkan, "vkCmdSetEvent"));
    vkCmdResetEvent       = reinterpret_cast<PFN_vkCmdResetEvent>(dlsym(libvulkan, "vkCmdResetEvent"));
    vkCmdWaitEvents       = reinterpret_cast<PFN_vkCmdWaitEvents>(dlsym(libvulkan, "vkCmdWaitEvents"));
    vkCmdPipelineBarrier  = reinterpret_cast<PFN_vkCmdPipelineBarrier>(dlsym(libvulkan, "vkCmdPipelineBarrier"));
    vkCmdBeginQuery       = reinterpret_cast<PFN_vkCmdBeginQuery>(dlsym(libvulkan, "vkCmdBeginQuery"));
    vkCmdEndQuery         = reinterpret_cast<PFN_vkCmdEndQuery>(dlsym(libvulkan, "vkCmdEndQuery"));
    vkCmdResetQueryPool   = reinterpret_cast<PFN_vkCmdResetQueryPool>(dlsym(libvulkan, "vkCmdResetQueryPool"));
    vkCmdWriteTimestamp   = reinterpret_cast<PFN_vkCmdWriteTimestamp>(dlsym(libvulkan, "vkCmdWriteTimestamp"));
    vkCmdCopyQueryPoolResults =
        reinterpret_cast<PFN_vkCmdCopyQueryPoolResults>(dlsym(libvulkan, "vkCmdCopyQueryPoolResults"));
    vkCmdPushConstants   = reinterpret_cast<PFN_vkCmdPushConstants>(dlsym(libvulkan, "vkCmdPushConstants"));
    vkCmdBeginRenderPass = reinterpret_cast<PFN_vkCmdBeginRenderPass>(dlsym(libvulkan, "vkCmdBeginRenderPass"));
    vkCmdNextSubpass     = reinterpret_cast<PFN_vkCmdNextSubpass>(dlsym(libvulkan, "vkCmdNextSubpass"));
    vkCmdEndRenderPass   = reinterpret_cast<PFN_vkCmdEndRenderPass>(dlsym(libvulkan, "vkCmdEndRenderPass"));
    vkCmdExecuteCommands = reinterpret_cast<PFN_vkCmdExecuteCommands>(dlsym(libvulkan, "vkCmdExecuteCommands"));
    vkDestroySurfaceKHR  = reinterpret_cast<PFN_vkDestroySurfaceKHR>(dlsym(libvulkan, "vkDestroySurfaceKHR"));
    vkGetPhysicalDeviceSurfaceSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
        dlsym(libvulkan, "vkGetPhysicalDeviceSurfaceSupportKHR"));
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
        dlsym(libvulkan, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
    vkGetPhysicalDeviceSurfaceFormatsKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
        dlsym(libvulkan, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
    vkGetPhysicalDeviceSurfacePresentModesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
        dlsym(libvulkan, "vkGetPhysicalDeviceSurfacePresentModesKHR"));
    vkCreateSwapchainKHR  = reinterpret_cast<PFN_vkCreateSwapchainKHR>(dlsym(libvulkan, "vkCreateSwapchainKHR"));
    vkDestroySwapchainKHR = reinterpret_cast<PFN_vkDestroySwapchainKHR>(dlsym(libvulkan, "vkDestroySwapchainKHR"));
    vkGetSwapchainImagesKHR =
        reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(dlsym(libvulkan, "vkGetSwapchainImagesKHR"));
    vkAcquireNextImageKHR = reinterpret_cast<PFN_vkAcquireNextImageKHR>(dlsym(libvulkan, "vkAcquireNextImageKHR"));
    vkQueuePresentKHR     = reinterpret_cast<PFN_vkQueuePresentKHR>(dlsym(libvulkan, "vkQueuePresentKHR"));
    vkGetPhysicalDeviceDisplayPropertiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPropertiesKHR>(
        dlsym(libvulkan, "vkGetPhysicalDeviceDisplayPropertiesKHR"));
    vkGetPhysicalDeviceDisplayPlanePropertiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR>(
        dlsym(libvulkan, "vkGetPhysicalDeviceDisplayPlanePropertiesKHR"));
    vkGetDisplayPlaneSupportedDisplaysKHR = reinterpret_cast<PFN_vkGetDisplayPlaneSupportedDisplaysKHR>(
        dlsym(libvulkan, "vkGetDisplayPlaneSupportedDisplaysKHR"));
    vkGetDisplayModePropertiesKHR =
        reinterpret_cast<PFN_vkGetDisplayModePropertiesKHR>(dlsym(libvulkan, "vkGetDisplayModePropertiesKHR"));
    vkCreateDisplayModeKHR = reinterpret_cast<PFN_vkCreateDisplayModeKHR>(dlsym(libvulkan, "vkCreateDisplayModeKHR"));
    vkGetDisplayPlaneCapabilitiesKHR =
        reinterpret_cast<PFN_vkGetDisplayPlaneCapabilitiesKHR>(dlsym(libvulkan, "vkGetDisplayPlaneCapabilitiesKHR"));
    vkCreateDisplayPlaneSurfaceKHR =
        reinterpret_cast<PFN_vkCreateDisplayPlaneSurfaceKHR>(dlsym(libvulkan, "vkCreateDisplayPlaneSurfaceKHR"));
    vkCreateSharedSwapchainsKHR =
        reinterpret_cast<PFN_vkCreateSharedSwapchainsKHR>(dlsym(libvulkan, "vkCreateSharedSwapchainsKHR"));

#ifdef VK_USE_PLATFORM_XLIB_KHR
    vkCreateXlibSurfaceKHR = reinterpret_cast<PFN_vkCreateXlibSurfaceKHR>(dlsym(libvulkan, "vkCreateXlibSurfaceKHR"));
    vkGetPhysicalDeviceXlibPresentationSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceXlibPresentationSupportKHR>(
        dlsym(libvulkan, "vkGetPhysicalDeviceXlibPresentationSupportKHR"));
#endif

#ifdef VK_USE_PLATFORM_XCB_KHR
    vkCreateXcbSurfaceKHR = reinterpret_cast<PFN_vkCreateXcbSurfaceKHR>(dlsym(libvulkan, "vkCreateXcbSurfaceKHR"));
    vkGetPhysicalDeviceXcbPresentationSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceXcbPresentationSupportKHR>(
        dlsym(libvulkan, "vkGetPhysicalDeviceXcbPresentationSupportKHR"));
#endif

#ifdef VK_USE_PLATFORM_WAYLAND_KHR
    vkCreateWaylandSurfaceKHR =
        reinterpret_cast<PFN_vkCreateWaylandSurfaceKHR>(dlsym(libvulkan, "vkCreateWaylandSurfaceKHR"));
    vkGetPhysicalDeviceWaylandPresentationSupportKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR>(
            dlsym(libvulkan, "vkGetPhysicalDeviceWaylandPresentationSupportKHR"));
#endif

#ifdef VK_USE_PLATFORM_MIR_KHR
    vkCreateMirSurfaceKHR = reinterpret_cast<PFN_vkCreateMirSurfaceKHR>(dlsym(libvulkan, "vkCreateMirSurfaceKHR"));
    vkGetPhysicalDeviceMirPresentationSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceMirPresentationSupportKHR>(
        dlsym(libvulkan, "vkGetPhysicalDeviceMirPresentationSupportKHR"));
#endif

#ifdef VK_USE_PLATFORM_ANDROID_KHR
    vkCreateAndroidSurfaceKHR =
        reinterpret_cast<PFN_vkCreateAndroidSurfaceKHR>(dlsym(libvulkan, "vkCreateAndroidSurfaceKHR"));
#endif

#ifdef VK_USE_PLATFORM_WIN32_KHR
    vkCreateWin32SurfaceKHR =
        reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(dlsym(libvulkan, "vkCreateWin32SurfaceKHR"));
    vkGetPhysicalDeviceWin32PresentationSupportKHR =
        reinterpret_cast<PFN_vkGetPhysicalDeviceWin32PresentationSupportKHR>(
            dlsym(libvulkan, "vkGetPhysicalDeviceWin32PresentationSupportKHR"));
#endif
#ifdef USE_DEBUG_EXTENTIONS
    vkCreateDebugReportCallbackEXT =
        reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(dlsym(libvulkan, "vkCreateDebugReportCallbackEXT"));
    vkDestroyDebugReportCallbackEXT =
        reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(dlsym(libvulkan, "vkDestroyDebugReportCallbackEXT"));
    vkDebugReportMessageEXT =
        reinterpret_cast<PFN_vkDebugReportMessageEXT>(dlsym(libvulkan, "vkDebugReportMessageEXT"));
#endif
    return 1;
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
