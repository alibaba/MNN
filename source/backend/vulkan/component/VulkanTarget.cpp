#include "VulkanTarget.hpp"
namespace MNN {

void VulkanTarget::onEnter(VkCommandBuffer buffer) {
    mBeginInfo.framebuffer = mContent.framebuffer->get();
    vkCmdBeginRenderPass(buffer, &mBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
}
void VulkanTarget::onExit(VkCommandBuffer buffer) {
    vkCmdEndRenderPass(buffer);
}

VulkanTarget::VulkanTarget() {
    auto beginInfo = &mBeginInfo;
    ::memset(beginInfo, 0, sizeof(VkRenderPassBeginInfo));
    beginInfo->sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    mMultisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    mMultisampleInfo.pNext = nullptr;
    mMultisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    mMultisampleInfo.sampleShadingEnable = VK_FALSE;
    mMultisampleInfo.minSampleShading = 0;
    mMultisampleInfo.pSampleMask = &mSampleMask;
    mMultisampleInfo.alphaToCoverageEnable = VK_FALSE;
    mMultisampleInfo.alphaToOneEnable = VK_FALSE;
}
VulkanTarget::~VulkanTarget() {
    // Do nothing
}

VkRenderPass VulkanTarget::pass() const {
    return mContent.pass->get();
}
VkExtent2D VulkanTarget::displaySize() const {
    VkExtent2D size;
    size.width = mContent.colors[0]->width();
    size.height = mContent.colors[0]->height();
    return size;
}

VulkanTarget* VulkanTarget::create(std::vector<SharedPtr<VulkanImage>> colors, SharedPtr<VulkanImage> depth) {
    int width = depth->width();
    int height = depth->height();
    auto target = new VulkanTarget;
    target->mClearValue.resize(1 + colors.size());
    auto beginInfo = &target->mBeginInfo;
    beginInfo->clearValueCount = target->mClearValue.size();
    beginInfo->pClearValues = target->mClearValue.data();
    target->mClearValue[colors.size()].depthStencil.depth = 1.0f;
    for (int i=0; i<colors.size(); ++i) {
        ::memset(target->mClearValue[i].color.float32, 0, 4 * sizeof(float));
    }
    auto depthFormat = depth->format();
    // Create render pass
    std::vector<VkAttachmentDescription> attachmentsDescriptionAll(1 + colors.size());
    std::vector<VkAttachmentReference> colourReferences(colors.size());
    for (int i=0; i<colors.size(); ++i) {
        auto& atta = attachmentsDescriptionAll[i];
        atta.format = colors[i]->format();
        atta.samples = VK_SAMPLE_COUNT_1_BIT;
        atta.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atta.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        atta.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atta.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atta.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        atta.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference& colourReference = colourReferences[i];
        colourReference.attachment = i;
        colourReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    auto& depthAttachment = attachmentsDescriptionAll[colors.size()];
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.format = depthFormat;

    VkAttachmentReference depthReference;
    depthReference.attachment = (uint32_t)colors.size();
    depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpassDescription;
    subpassDescription.flags = 0;
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.inputAttachmentCount = 0;
    subpassDescription.pInputAttachments = nullptr;
    subpassDescription.colorAttachmentCount = (uint32_t)colourReferences.size();
    subpassDescription.pColorAttachments = colourReferences.data();
    subpassDescription.pResolveAttachments = nullptr;
    subpassDescription.pDepthStencilAttachment = &depthReference;
    subpassDescription.preserveAttachmentCount = 0;
    subpassDescription.pPreserveAttachments = nullptr;

    VkRenderPassCreateInfo renderPassCreateInfo;
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.pNext = nullptr;
    renderPassCreateInfo.flags = 0;
    renderPassCreateInfo.attachmentCount = (uint32_t)attachmentsDescriptionAll.size();
    renderPassCreateInfo.pAttachments = attachmentsDescriptionAll.data();
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpassDescription;
    renderPassCreateInfo.dependencyCount = 0;
    renderPassCreateInfo.pDependencies = nullptr;

    const auto& dev = depth->device();
    target->mContent.pass = VulkanRenderPass::create(dev, &renderPassCreateInfo);
    target->mContent.colors = colors;
    target->mContent.depth = depth;
    target->mAttachments.resize((1 + colors.size()));
    target->mAttachments[colors.size()] = depth->view();
    for (int i=0; i<colors.size(); ++i) {
        target->mAttachments[i] = colors[i]->view();
    }
    VkFramebufferCreateInfo& fbCreateInfo = target->mFbCreateInfo;
    fbCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
    fbCreateInfo.pNext = nullptr,
    fbCreateInfo.renderPass = target->mContent.pass->get(),
    fbCreateInfo.pAttachments = target->mAttachments.data(),
    fbCreateInfo.attachmentCount = (uint32_t)target->mAttachments.size(),
    fbCreateInfo.width = width,
    fbCreateInfo.height = height,
    fbCreateInfo.layers = 1;
    target->mContent.framebuffer = VulkanFramebuffer::create(dev, &fbCreateInfo);
    target->mBeginInfo.renderPass = target->mContent.pass->get();
    beginInfo->renderPass = target->mContent.pass->get();
    beginInfo->renderArea.offset.x = 0;
    beginInfo->renderArea.offset.y = 0;
    beginInfo->renderArea.extent.width = width;
    beginInfo->renderArea.extent.height = height;
    auto& scissor = target->mScissor;
    scissor.offset.x = 0; scissor.offset.y = 0;
    scissor.extent.width = width;
    scissor.extent.height = height;
    auto& viewport = target->mViewPortState;
    ::memset(&viewport, 0, sizeof(VkPipelineViewportStateCreateInfo));
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.scissorCount = 1;
    viewport.pScissors = &scissor;
    
    auto& view = target->mViewPort;
    view.x = 0; view.y = 0; view.maxDepth = 1.0f; view.minDepth = 0.0f;
    view.width = width; view.height = height;
    viewport.viewportCount = 1;
    viewport.pViewports = &view;

    return target;
}

void VulkanTarget::writePipelineInfo(VkGraphicsPipelineCreateInfo& info) const {
    auto pColor = (VkPipelineColorBlendStateCreateInfo*)info.pColorBlendState;
    auto pDepth = (VkPipelineColorBlendAttachmentState*)info.pDepthStencilState;
    pColor->attachmentCount = mContent.colors.size();
    
    info.renderPass = pass();
    info.pViewportState = &mViewPortState;

    info.pMultisampleState = &mMultisampleInfo;
}
};
