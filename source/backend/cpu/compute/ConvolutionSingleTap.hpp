#ifndef ConvolutionSingleTap_hpp
#define ConvolutionSingleTap_hpp

#include <memory>
#include "core/Execution.hpp"

namespace flatbuffers {
class FlatBufferBuilder;
}

namespace MNN {
struct Convolution2DCommon;

// Selects a 1x1 executor when only the center tap contributes.
// Keep both packed kernels so resize can select either path without repacking.
// Clones share their weights and the reduced op metadata.
class ConvolutionSingleTap : public Execution {
public:
    static bool liveTaps(const Convolution2DCommon* common, const Tensor* input, const Tensor* output,
                         int* ky, int* kx);
    ConvolutionSingleTap(Backend* backend, const Convolution2DCommon* common, int ky, int kx,
                         std::shared_ptr<Execution> full, std::shared_ptr<flatbuffers::FlatBufferBuilder> reducedOp,
                         std::shared_ptr<Execution> reduced);
    bool onClone(Backend* backend, const Op* op, Execution** dst) override;
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    const Convolution2DCommon* mCommon;
    const int mKy;
    const int mKx;
    std::shared_ptr<Execution> mFull;
    // Owns the 1x1 op mReduced was built from; the execution keeps pointers
    // into it, so it is declared first and destroyed last.
    std::shared_ptr<flatbuffers::FlatBufferBuilder> mReducedOp;
    std::shared_ptr<Execution> mReduced;
    Execution* mActive = nullptr;
};

} // namespace MNN

#endif // ConvolutionSingleTap_hpp
