#ifndef HexagonCommand_hpp
#define HexagonCommand_hpp

#include <vector>
#include <tuple>
#include <memory>
#include <map>
#include <utility>
#include <string.h>
#include "core/BufferAllocator.hpp"

namespace MNN {
class HexagonBackend;
class Tensor;

class HexagonCommand {
public:
    HexagonCommand() = default;
    HexagonCommand(const HexagonCommand&) = delete;
    HexagonCommand& operator=(const HexagonCommand&) = delete;
    HexagonCommand(HexagonCommand&& other) noexcept;
    HexagonCommand& operator=(HexagonCommand&& other) noexcept;
    ~HexagonCommand();

    void build(HexagonBackend* backend, int opType, const void* param, int paramSize,
               const std::vector<std::pair<int, int>>& inputFdOffsets,
               const std::vector<std::pair<int, int>>& outputFdOffsets,
               const std::vector<Tensor*>& inputs = {},
               const std::vector<Tensor*>& outputs = {});

    int execute(bool forceCopy = false);

    void addTensorMap(Tensor* tensor, int index) {
        mInputTensorIndexes.push_back({tensor, index});
        mInputDevicePtrs.emplace_back(-1, -1);
    }
    void setInputTensor(Tensor* tensor, int index);

    void* getParam();

    std::vector<std::pair<Tensor*, int>> mInputTensorIndexes;
    std::vector<std::pair<Tensor*, int>> mOutputTensorIndexes;

private:
    friend class HexagonBackend;
    void encode(const std::vector<std::pair<int, int>>& inputFdOffsets,
                const std::vector<std::pair<int, int>>& outputFdOffsets);
    HexagonBackend* mBackend = nullptr;
    MemChunk mCommandChunk;
    size_t mCmdSize = 0;
    size_t mCommandCapacity = 0;
    int mOpType = 0;
    std::vector<uint8_t> mParamData;
    std::vector<std::pair<int, int>> mInputFdOffsets;
    std::vector<std::pair<int, int>> mOutputFdOffsets;
    int mLastQueuedSerial = 0;
    bool mDirty = true;
    std::vector<std::pair<int, int>> mInputDevicePtrs;
    std::vector<std::pair<int, int>> mOutputDevicePtrs;
};

}

#endif
