//
//  VulkanLSTM.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanLSTM_hpp
#define VulkanLSTM_hpp
#include "VulkanBasicExecution.hpp"

namespace MNN {

class LSTMChannel {
public:
    LSTMChannel(const VulkanPipeline* vulkanLSTMPipeline, VulkanBackend* vkbackend, const int channel);
    ~LSTMChannel();

    ErrorCode encodeImpl(std::shared_ptr<VulkanBuffer>& gates, std::shared_ptr<VulkanBuffer>& cells,
                         std::shared_ptr<VulkanBuffer>& weightH, std::shared_ptr<VulkanBuffer>& bias,
                         std::shared_ptr<VulkanBuffer>& out, const VulkanCommandPool::Buffer* cmdBuffer, const int ow);

private:
    int mChannel;
    const VulkanPipeline* mVulkanLSTMPipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mDescriptorSet;
    std::shared_ptr<VulkanBuffer> mParamBuffer;
};

class VulkanLSTM : public VulkanBasicExecution {
public:
    VulkanLSTM(const LSTM* lstm, Backend* bn);
    virtual ~VulkanLSTM();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    ErrorCode _resize(const Tensor* input, const Tensor* output);

    const LSTM* mLSTM;
    VulkanBackend* mVKbackend;

    // gates
    const VulkanPipeline* mVulkanLSTMGatePipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mGateDescriptorSet;
    std::shared_ptr<VulkanBuffer> mGateParamBuffer;
    // nchw -> nc4hw4
    const VulkanPipeline* mVulkanLSTMSavePipeline;
    std::shared_ptr<VulkanPipeline::DescriptorSet> mSaveDescriptorSet;
    std::shared_ptr<VulkanBuffer> mSaveParamBuffer;
    // channel loop
    std::vector<std::shared_ptr<LSTMChannel>> mLSTMChannels;
    const VulkanPipeline* mVulkanLSTMPipeline;

    std::shared_ptr<VulkanBuffer> mWeightI;
    std::shared_ptr<VulkanBuffer> mWeightH;
    std::shared_ptr<VulkanBuffer> mBias;

    std::shared_ptr<VulkanBuffer> mGate;
    std::shared_ptr<VulkanBuffer> mCell;
    std::shared_ptr<VulkanBuffer> mOutputTemp;
};
} // namespace MNN
#endif
