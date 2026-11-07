#include "ConvolutionSingleTap.hpp"
#include "core/ConvolutionCommon.hpp"
#include <utility>

namespace MNN {
namespace {

// The single live tap along one axis, or -1 when there is not exactly one
// or it does not sample the input at the output position.
static int liveTap(int kernel, int dilate, int stride, int pad, int inSize, int outSize) {
    int live = -1;
    for (int t = 0; t < kernel; ++t) {
        const int first = t * dilate - pad;
        const int last  = (outSize - 1) * stride + first;
        if (last >= 0 && first < inSize) {
            if (live >= 0) {
                return -1;
            }
            live = t;
        }
    }
    if (live < 0 || live * dilate - pad != 0) {
        return -1;
    }
    // An unpadded 1x1 convolution with the same stride must produce the
    // same output size.
    if ((inSize - 1) / stride + 1 != outSize) {
        return -1;
    }
    return live;
}

} // namespace

bool ConvolutionSingleTap::liveTaps(const Convolution2DCommon* common, const Tensor* input, const Tensor* output,
                                    int* ky, int* kx) {
    if (common->kernelX() * common->kernelY() <= 1 || input->batch() != output->batch()) {
        return false;
    }
    const auto pad = ConvolutionCommon::convolutionPad(input, output, common);
    *kx = liveTap(common->kernelX(), common->dilateX(), common->strideX(), pad.first, input->width(), output->width());
    *ky = liveTap(common->kernelY(), common->dilateY(), common->strideY(), pad.second,
                  input->height(), output->height());
    return *kx >= 0 && *ky >= 0;
}

ConvolutionSingleTap::ConvolutionSingleTap(Backend* backend, const Convolution2DCommon* common, int ky, int kx,
                                         std::shared_ptr<Execution> full,
                                         std::shared_ptr<flatbuffers::FlatBufferBuilder> reducedOp,
                                         std::shared_ptr<Execution> reduced)
    : Execution(backend), mCommon(common), mKy(ky), mKx(kx), mFull(std::move(full)),
      mReducedOp(std::move(reducedOp)), mReduced(std::move(reduced)) {
    mValid = mFull && mFull->valid() && mReduced && mReduced->valid();
}

bool ConvolutionSingleTap::onClone(Backend* backend, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    auto reducedOp = flatbuffers::GetRoot<Op>(mReducedOp->GetBufferPointer());
    if (nullptr == dst) {
        return mFull->onClone(backend, op, nullptr) && mReduced->onClone(backend, reducedOp, nullptr);
    }
    Execution* full = nullptr;
    const bool fullCloned = mFull->onClone(backend, op, &full);
    std::shared_ptr<Execution> fullOwner(full);
    if (!fullCloned || !full || !full->valid()) {
        return false;
    }
    Execution* reduced = nullptr;
    const bool reducedCloned = mReduced->onClone(backend, reducedOp, &reduced);
    std::shared_ptr<Execution> reducedOwner(reduced);
    if (!reducedCloned || !reduced || !reduced->valid()) {
        return false;
    }
    *dst = new ConvolutionSingleTap(backend, op->main_as_Convolution2D()->common(), mKy, mKx,
                                  std::move(fullOwner), mReducedOp, std::move(reducedOwner));
    return true;
}

ErrorCode ConvolutionSingleTap::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (!mValid) {
        return OUT_OF_MEMORY;
    }
    int ky = -1, kx = -1;
    const bool reducible = liveTaps(mCommon, inputs[0], outputs[0], &ky, &kx) && ky == mKy && kx == mKx;
    mActive = reducible ? mReduced.get() : mFull.get();
    return mActive->onResize(inputs, outputs);
}

ErrorCode ConvolutionSingleTap::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return mActive->onExecute(inputs, outputs);
}

} // namespace MNN
