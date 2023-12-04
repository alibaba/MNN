#ifndef VulkanTarget_hpp
#define VulkanTarget_hpp
#include "VulkanImage.hpp"
#include "VulkanMemoryPool.hpp"
#include "VulkanRenderPass.hpp"
#include "VulkanFence.hpp"
#include "VulkanCommandPool.hpp"
#include "VulkanSemaphore.hpp"
#include "VulkanFramebuffer.hpp"
namespace MNN {
class VulkanTarget : public RefCount {
public:
    virtual ~ VulkanTarget();
    
    void onEnter(VkCommandBuffer buffer);
    void onExit(VkCommandBuffer buffer);
    
    // For Default Target
    static VulkanTarget* create(std::vector<SharedPtr<VulkanImage>> colors, SharedPtr<VulkanImage> depth);
    
    void writePipelineInfo(VkGraphicsPipelineCreateInfo& info) const;
    struct Content {
        std::vector<SharedPtr<VulkanImage>> colors;
        SharedPtr<VulkanImage> depth;
        SharedPtr<VulkanRenderPass> pass;
        SharedPtr<VulkanFramebuffer> framebuffer;
    };
    VkRenderPass pass() const;
    VkExtent2D displaySize() const;
    const Content& content() const {
        return mContent;
    }
private:
    VulkanTarget();
    std::vector<VkImageView> mAttachments;
    std::vector<VkClearValue> mClearValue;
    VkRenderPassBeginInfo mBeginInfo;
    Content mContent;
    VkPipelineViewportStateCreateInfo mViewPortState;
    VkViewport mViewPort;
    VkRect2D mScissor;
    // TODO: Support multi sample
    VkSampleMask mSampleMask = ~0u;
    VkPipelineMultisampleStateCreateInfo mMultisampleInfo;
    VkFramebufferCreateInfo mFbCreateInfo;
};
};

#endif
