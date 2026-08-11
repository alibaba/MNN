#ifndef HexagonLoop_hpp
#define HexagonLoop_hpp

#include <vector>

#include "core/BufferAllocator.hpp"
#include "HexagonExecution.hpp"

namespace MNN {

struct LoopParam;

class HexagonLoop : public HexagonExecution {
public:
    virtual ~HexagonLoop();

    static HexagonLoop* create(Backend* backend, const Op* op);

private:
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    explicit HexagonLoop(Backend* backend, const LoopParam* loop);

    const LoopParam* mLoop = nullptr;

    std::vector<Tensor*> mStack;
    BufferAllocator* mAllocator = nullptr;

    // For invalid input -> output zero
    MemChunk mZeroChunk;


    // For init zeros param structs
    MemChunk mInitZeroParamChunk;

    // For initCommand copy (single region scratch)
    MemChunk mInitRegionChunk;

    int mBytes = 2;
    int mPack = 4;
    int mLoopNumber = 0;

    struct InitCopy {
        int dstIndex = -1;
        int srcIndex = -1;
        int size[3] = {1, 1, 1};
        int srcStride[3] = {1, 1, 1};
        int dstStride[3] = {1, 1, 1};
        int srcOffset = 0;
        int dstOffset = 0;
    };

    std::vector<int> mInitZeroTensorIndexes;
    std::vector<InitCopy> mInitCopyCommands;

    // Command info (RegionCommand order: [z,y,x])
    int mCmdIndexes[2] = {-1, -1};
    int mCmdIterIndexes[2] = {-1, -1};
    int mCmdSteps[2] = {0, 0};

    int mCmdViewOffset[2] = {0, 0};
    int mCmdViewStride[2][3] = {{1, 1, 1}, {1, 1, 1}};

    int mCmdSizeZYX[3] = {1, 1, 1};

    std::vector<std::shared_ptr<HexagonCommand>> mInitZeroCmds;
    std::vector<std::shared_ptr<HexagonCommand>> mInitCopyCmds;
};

} // namespace MNN

#endif
