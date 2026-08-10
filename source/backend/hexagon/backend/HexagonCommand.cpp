#include "HexagonCommand.hpp"
#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "schema/current/Command_generated.h"
#include "core/TensorUtils.hpp"
#include "core/BufferAllocator.hpp"

namespace MNN {

HexagonCommand::HexagonCommand(HexagonCommand&& other) noexcept {
    *this = std::move(other);
}

HexagonCommand& HexagonCommand::operator=(HexagonCommand&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (mBackend != nullptr && mCommandChunk.first != nullptr) {
        if (mLastQueuedSerial == mBackend->commandSerial()) {
            mBackend->flushCommand();
        }
        mBackend->freeCommandSlot(mCommandChunk);
    }
    mBackend = other.mBackend;
    mCommandChunk = other.mCommandChunk;
    mCmdSize = other.mCmdSize;
    mCommandCapacity = other.mCommandCapacity;
    mOpType = other.mOpType;
    mParamData = std::move(other.mParamData);
    mInputFdOffsets = std::move(other.mInputFdOffsets);
    mOutputFdOffsets = std::move(other.mOutputFdOffsets);
    mLastQueuedSerial = other.mLastQueuedSerial;
    mDirty = other.mDirty;
    mInputTensorIndexes = std::move(other.mInputTensorIndexes);
    mOutputTensorIndexes = std::move(other.mOutputTensorIndexes);
    mInputDevicePtrs = std::move(other.mInputDevicePtrs);
    mOutputDevicePtrs = std::move(other.mOutputDevicePtrs);

    other.mBackend = nullptr;
    other.mCommandChunk = MemChunk();
    other.mCmdSize = 0;
    other.mCommandCapacity = 0;
    other.mOpType = 0;
    other.mLastQueuedSerial = 0;
    other.mDirty = true;
    return *this;
}

HexagonCommand::~HexagonCommand() {
    if (mBackend != nullptr && mCommandChunk.first != nullptr) {
        if (mLastQueuedSerial == mBackend->commandSerial()) {
            mBackend->flushCommand();
        }
        mBackend->freeCommandSlot(mCommandChunk);
        mCommandChunk = MemChunk();
        mCommandCapacity = 0;
    }
}

void HexagonCommand::build(HexagonBackend* backend, int opType, const void* param, int paramSize,
            const std::vector<std::pair<int, int>>& inputFdOffsets,
            const std::vector<std::pair<int, int>>& outputFdOffsets,
            const std::vector<Tensor*>& inputs,
            const std::vector<Tensor*>& outputs) {
    mInputTensorIndexes.clear();
    mOutputTensorIndexes.clear();
    mInputDevicePtrs.clear();
    mOutputDevicePtrs.clear();

    if (mBackend != nullptr && mCommandChunk.first != nullptr) {
        mBackend->freeCommandSlot(mCommandChunk);
        mCommandChunk = MemChunk();
        mCommandCapacity = 0;
    }
    mBackend = backend;
    mLastQueuedSerial = 0;
    mDirty = true;
    mOpType = opType;
    mParamData.clear();
    if (paramSize > 0 && param != nullptr) {
        mParamData.resize(paramSize);
        ::memcpy(mParamData.data(), param, paramSize);
    }
    mInputFdOffsets = inputFdOffsets;
    mOutputFdOffsets = outputFdOffsets;
    encode(mInputFdOffsets, mOutputFdOffsets);

    std::vector<uint8_t> usedInputs(inputFdOffsets.size(), 0);
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] == nullptr) {
            continue;
        }
        auto dev = HexagonBackend::getDevicePtr(inputs[i]);
        for (size_t j = 0; j < inputFdOffsets.size(); ++j) {
            if (usedInputs[j]) {
                continue;
            }
            if (dev.first == inputFdOffsets[j].first && dev.second == inputFdOffsets[j].second) {
                mInputTensorIndexes.emplace_back(inputs[i], (int)j);
                mInputDevicePtrs.emplace_back(dev);
                usedInputs[j] = 1;
                break;
            }
        }
    }
    std::vector<uint8_t> usedOutputs(outputFdOffsets.size(), 0);
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i] == nullptr) {
            continue;
        }
        auto dev = HexagonBackend::getDevicePtr(outputs[i]);
        for (size_t j = 0; j < outputFdOffsets.size(); ++j) {
            if (usedOutputs[j]) {
                continue;
            }
            if (dev.first == outputFdOffsets[j].first && dev.second == outputFdOffsets[j].second) {
                mOutputTensorIndexes.emplace_back(outputs[i], (int)j);
                mOutputDevicePtrs.emplace_back(dev);
                usedOutputs[j] = 1;
                break;
            }
        }
    }
}

void HexagonCommand::encode(const std::vector<std::pair<int, int>>& inputFdOffsets,
                            const std::vector<std::pair<int, int>>& outputFdOffsets) {
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> inputTensors;
    inputTensors.reserve(inputFdOffsets.size());
    for (size_t i = 0; i < inputFdOffsets.size(); i++) {
        auto tensor = DSPCOMMAND::CreateTensor(builder, inputFdOffsets[i].first, inputFdOffsets[i].second, 0);
        inputTensors.push_back(tensor);
    }

    std::vector<flatbuffers::Offset<DSPCOMMAND::Tensor>> outputTensors;
    outputTensors.reserve(outputFdOffsets.size());
    for (size_t i = 0; i < outputFdOffsets.size(); i++) {
        auto tensor = DSPCOMMAND::CreateTensor(builder, outputFdOffsets[i].first, outputFdOffsets[i].second, 0);
        outputTensors.push_back(tensor);
    }

    flatbuffers::Offset<flatbuffers::Vector<int32_t>> paramsOffset;
    if (!mParamData.empty()) {
        paramsOffset = builder.CreateVector(reinterpret_cast<const int32_t*>(mParamData.data()),
                                            mParamData.size() / sizeof(int32_t));
    }

    auto command = DSPCOMMAND::CreateCommand(builder, mOpType, builder.CreateVector(inputTensors),
                                             builder.CreateVector(outputTensors), paramsOffset);
    builder.Finish(command);

    mCmdSize = builder.GetSize();
    if (mCmdSize > mCommandCapacity) {
        if (mCommandChunk.first != nullptr) {
            mBackend->freeCommandSlot(mCommandChunk);
            mCommandChunk = MemChunk();
            mCommandCapacity = 0;
        }
        mCommandChunk = mBackend->allocCommandSlot((int)mCmdSize);
        MNN_ASSERT(mCommandChunk.first != nullptr);
        mCommandCapacity = mCmdSize;
    }
    auto cmdPtr = HexagonBackend::getPtr(mCommandChunk);
    ::memcpy(cmdPtr, builder.GetBufferPointer(), mCmdSize);
}

int HexagonCommand::execute(bool forceCopy) {
    MNN_ASSERT(mCommandChunk.first != nullptr);
    bool needPatch = mInputDevicePtrs.size() != mInputTensorIndexes.size() ||
                     mOutputDevicePtrs.size() != mOutputTensorIndexes.size();
    if (!needPatch) {
        for (size_t i = 0; i < mInputTensorIndexes.size(); ++i) {
            auto dev = HexagonBackend::getDevicePtr(mInputTensorIndexes[i].first);
            if (dev != mInputDevicePtrs[i]) {
                needPatch = true;
                break;
            }
        }
    }
    if (!needPatch) {
        for (size_t i = 0; i < mOutputTensorIndexes.size(); ++i) {
            auto dev = HexagonBackend::getDevicePtr(mOutputTensorIndexes[i].first);
            if (dev != mOutputDevicePtrs[i]) {
                needPatch = true;
                break;
            }
        }
    }
    if (needPatch) {
        {
            auto cmdPtr = HexagonBackend::getPtr(mCommandChunk);
            auto command = flatbuffers::GetRoot<DSPCOMMAND::Command>(cmdPtr);
            auto params = command->params();
            if (params != nullptr && params->size() > 0) {
                const size_t paramBytes = params->size() * sizeof(int32_t);
                mParamData.resize(paramBytes);
                ::memcpy(mParamData.data(), params->data(), paramBytes);
            }
        }
        auto inputFdOffsets = mInputFdOffsets;
        auto outputFdOffsets = mOutputFdOffsets;
        mInputDevicePtrs.resize(mInputTensorIndexes.size());
        for (size_t i = 0; i < mInputTensorIndexes.size(); ++i) {
            auto tensor = mInputTensorIndexes[i].first;
            int idx = mInputTensorIndexes[i].second;
            auto dev = HexagonBackend::getDevicePtr(tensor);
            if (idx >= 0 && idx < inputFdOffsets.size()) {
                inputFdOffsets[idx] = dev;
            }
            mInputDevicePtrs[i] = dev;
            mDirty = true;
        }

        mOutputDevicePtrs.resize(mOutputTensorIndexes.size());
        for (size_t i = 0; i < mOutputTensorIndexes.size(); ++i) {
            auto tensor = mOutputTensorIndexes[i].first;
            int idx = mOutputTensorIndexes[i].second;
            auto dev = HexagonBackend::getDevicePtr(tensor);
            if (idx >= 0 && idx < outputFdOffsets.size()) {
                outputFdOffsets[idx] = dev;
            }
            mOutputDevicePtrs[i] = dev;
            mDirty = true;
        }
        encode(inputFdOffsets, outputFdOffsets);
    }
    const int serial = mBackend->commandSerial();
    const bool needCopy = forceCopy || mLastQueuedSerial == serial;
    const bool dirty = mDirty || needCopy;
    for (const auto& output : mOutputTensorIndexes) {
        mBackend->markHexagonOutput(output.first);
    }
    mBackend->pushCommand(mCommandChunk, (int)mCmdSize, needCopy, dirty);
    mDirty = false;
    mLastQueuedSerial = mBackend->commandSerial();
    return 0;
}

void* HexagonCommand::getParam() {
    if (mCommandChunk.first == nullptr) return nullptr;
    auto cmdPtr = HexagonBackend::getPtr(mCommandChunk);
    auto command = flatbuffers::GetRoot<DSPCOMMAND::Command>(cmdPtr);
    auto params = command->params();
    if (params && params->size() > 0) {
        return (void*)params->data();
    }
    return nullptr;
}

void HexagonCommand::setInputTensor(Tensor* tensor, int index) {
    for (size_t i = 0; i < mInputTensorIndexes.size(); ++i) {
        if (mInputTensorIndexes[i].second == index) {
            mInputTensorIndexes[i].first = tensor;
            if (i < mInputDevicePtrs.size()) {
                mInputDevicePtrs[i] = std::make_pair(-1, -1);
            }
            mDirty = true;
            return;
        }
    }
    addTensorMap(tensor, index);
    mDirty = true;
}

}
