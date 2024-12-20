//
//  CPUStft.hpp
//  MNN
//
//  Created by MNN on 2024/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_BUILD_AUDIO
#ifndef CPUStft_hpp
#define CPUStft_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUStft : public Execution {
public:
    CPUStft(Backend *backend, int nfft, int hop_length, bool abs);
    virtual ~CPUStft() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    int mNfft, mHopLength;
    bool mAbs;
    Tensor mTmpFrames;
};

} // namespace MNN

#endif /* CPUStft.hpp */
#endif // MNN_BUILD_AUDIO