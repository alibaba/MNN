#include "VulkanLoop.hpp"
#include "VulkanBinary.hpp"

namespace MNN {

std::string getMidName(const Op* op) {
    std::string mid = "";
    if (op->type() == OpType_Eltwise) {
        if (op->main_as_Eltwise()->coeff() != nullptr) {
            // Don't support
            return "";
        }
        switch (op->main_as_Eltwise()->type()) {
            case EltwiseType_SUB:
                mid = "SUB";
                break;
            case EltwiseType_MAXIMUM:
                mid = "VMAX";
                break;
            case EltwiseType_PROD:
                mid = "MUL";
                break;
            case EltwiseType_SUM:
                mid = "ADD";
                break;
            default:
                break;
        }
    } else if (op->type() == OpType_BinaryOp) {
        switch (op->main_as_BinaryOp()->opType()) {
            case BinaryOpOperation_ADD:
                mid = "ADD";
                break;
            case BinaryOpOperation_SUB:
                mid = "SUB";
                break;
            case BinaryOpOperation_MAXIMUM:
                mid = "VMAX";
                break;
            case BinaryOpOperation_MINIMUM:
                mid = "VMIN";
                break;
            case BinaryOpOperation_MUL:
                mid = "MUL";
                break;
            case BinaryOpOperation_POW:
                mid = "POW";
                break;
            case BinaryOpOperation_SquaredDifference:
                mid = "SQUDIFF";
                break;
            case BinaryOpOperation_DIV:
            case BinaryOpOperation_REALDIV:
                mid = "DIV";
                break;
            default:
                break;
        }
    }
    return mid;
}

static void _setTensorStack(std::vector<Tensor*>& result, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const LoopParam* loop) {
    if (loop->inputIndexes() != nullptr) {
        for (int i=0; i<loop->inputIndexes()->size(); ++i) {
            result[loop->inputIndexes()->data()[i]] = inputs[i];
        }
    }
    for (int i=0; i<loop->outputIndexes()->size(); ++i) {
        result[loop->outputIndexes()->data()[i]] = outputs[i];
    }
}

struct BinaryBroadCastInfo {
    ivec4 srcview0;
    ivec4 srcview1;
    ivec4 dstview;
    ivec4 size;
};

class VulkanBinaryBroadCast : public VulkanBasicExecution {
public:
    VulkanBinaryBroadCast(const LoopParam* loop, Backend *bn, bool isInt) : VulkanBasicExecution(bn) {
        mLoop = loop;
        auto vkbackend = static_cast<VulkanBackend*>(bn);

        std::vector<VkDescriptorType> types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };

        std::string shaderName = "glsl_binary_blit_" + getMidName(mLoop->commands()->GetAs<RegionCommand>(0)->op()) + "_comp";

        mLoopPipeline = vkbackend->getPipeline(shaderName, types);
        mDescriptorSet.reset(mLoopPipeline->createSet());

        mGpuLoopParam.reset(new VulkanBuffer(vkbackend->getMemoryPool(), false, sizeof(BinaryBroadCastInfo), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
        mTensors.resize(mLoop->tensorNumber());
    }

    virtual ~VulkanBinaryBroadCast() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *>& inputs, const std::vector<Tensor *>& outputs,
                               const VulkanCommandPool::Buffer* cmdBuffer) override {
        _setTensorStack(mTensors, inputs, outputs, mLoop);
        auto cmd = mLoop->commands()->GetAs<RegionCommand>(0);
        auto size = cmd->size()->data();
        auto vkBn = static_cast<VulkanBackend*>(backend());
        auto srcStride0 = cmd->view()->GetAs<View>(1)->stride()->data();
        auto srcStride1 = cmd->view()->GetAs<View>(2)->stride()->data();
        auto dstStride = cmd->view()->GetAs<View>(0)->stride()->data();
        int totalSize = size[0] * size[1] * size[2];
        auto param = reinterpret_cast<BinaryBroadCastInfo*>(mGpuLoopParam->map());
        for (int i=0; i<3; ++i) {
            param->size[i] = size[i];
            param->srcview0[i] = srcStride0[i];
            param->srcview1[i] = srcStride1[i];
            param->dstview[i] = dstStride[i];
        }
        param->srcview0[3] = cmd->view()->GetAs<View>(1)->offset();
        param->srcview1[3] = cmd->view()->GetAs<View>(2)->offset();
        param->dstview[3] = cmd->view()->GetAs<View>(0)->offset();
        param->size[3] = size[0] * size[1] * size[2];
        mGpuLoopParam->unmap();
        auto output = mTensors[cmd->indexes()->data()[0]];
        auto input0 = mTensors[cmd->indexes()->data()[1]];
        auto input1 = mTensors[cmd->indexes()->data()[2]];

        {
            int bufferSizeSource0 = sizeof(float);
            for (int i=0; i<input0->dimensions(); ++i) {
                bufferSizeSource0 *= input0->length(i);
            }
            mInput0.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, bufferSizeSource0, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
            mInput0.convert.reset(new VulkanImageConverter(vkBn));
        }
        {
            int bufferSizeSource1 = sizeof(float);
            for (int i=0; i<input1->dimensions(); ++i) {
                bufferSizeSource1 *= input1->length(i);
            }
            mInput1.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, bufferSizeSource1, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
            mInput1.convert.reset(new VulkanImageConverter(vkBn));
        }
        {
            int bufferSizeOutput = sizeof(float);
            for (int i=0; i<output->dimensions(); ++i) {
                bufferSizeOutput *= output->length(i);
            }
            mOutput.buffer.reset(new VulkanBuffer(vkBn->getDynamicMemoryPool(), false, bufferSizeOutput, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
            mOutput.convert.reset(new VulkanImageConverter(vkBn));
        }
        mInput0.convert->encodeTensorToBuffer(input0, mInput0.buffer->buffer(), mInput0.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(inputs[0]), cmdBuffer);
        mInput1.convert->encodeTensorToBuffer(input1, mInput1.buffer->buffer(), mInput1.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(inputs[1]), cmdBuffer);

        mDescriptorSet->writeBuffer(mOutput.buffer->buffer(), 0, mOutput.buffer->size());
        mDescriptorSet->writeBuffer(mInput0.buffer->buffer(), 1, mInput0.buffer->size());
        mDescriptorSet->writeBuffer(mInput1.buffer->buffer(), 2, mInput1.buffer->size());
        mDescriptorSet->writeBuffer(mGpuLoopParam->buffer(), 3, mGpuLoopParam->size());

        cmdBuffer->barrierSource(mInput0.buffer->buffer(), 0, mInput0.buffer->size());
        cmdBuffer->barrierSource(mInput1.buffer->buffer(), 0, mInput1.buffer->size());

        mLoopPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(totalSize,256), 1, 1);

        cmdBuffer->barrierSource(mOutput.buffer->buffer(), 0, mOutput.buffer->size());
        mOutput.convert->encodeBufferToTensor(mOutput.buffer->buffer(), output, mOutput.buffer->size(), 0, VulkanImageConverter::getTensorLinearFormat(outputs[0]), cmdBuffer);
        mInput0.buffer->release();
        mInput1.buffer->release();
        mOutput.buffer->release();

        return NO_ERROR;
    }

private:
    const LoopParam* mLoop;
    const VulkanPipeline* mLoopPipeline;
    std::shared_ptr<VulkanBuffer> mGpuLoopParam;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
    std::vector<Tensor*> mTensors;
    struct ConvertInfo {
        std::shared_ptr<VulkanImageConverter> convert;
        std::shared_ptr<VulkanBuffer> buffer;
    };
    ConvertInfo mInput0;
    ConvertInfo mInput1;
    ConvertInfo mOutput;
};

VulkanBasicExecution* VulkanLoop::create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const Op* op, Backend* bn) {
    auto loop = op->main_as_LoopParam();
    if (nullptr == loop || loop->commands() == nullptr) {
        return nullptr;
    }
    if (nullptr != loop->initCommand()) {
        return nullptr;
    }

    if (1 == loop->commands()->size()) {
        auto cmd = loop->commands()->GetAs<RegionCommand>(0);
        auto subop = cmd->op();
        if (OpType_BinaryOp == subop->type() && cmd->fuse() < 0 && 1 == loop->loopNumber()) {
            std::string shaderMidName = getMidName(loop->commands()->GetAs<RegionCommand>(0)->op());
            if (shaderMidName.empty()) {
                return nullptr;
            }
            bool isInt = inputs[1]->getType().code == halide_type_int;
            if (isInt) {
                return nullptr;
            }
            return new VulkanBinaryBroadCast(loop, bn, isInt);
        }
    }
    return nullptr;
}

class VulkanLoopCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return VulkanLoop::create(inputs, outputs, op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_While, new VulkanLoopCreator);
    return true;
}();

}