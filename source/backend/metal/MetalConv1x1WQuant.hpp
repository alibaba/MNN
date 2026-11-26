//
//  MetalConv1x1WQuant.hpp
//  MNN
//
//  Weight-quantized conv1x1 executor: packed W2/W3/W4/W8 residency, decode GEMV
//  and quant prefill routing, prefill epilogues and split-K scratch.
//

#ifndef MetalConv1x1WQuant_hpp
#define MetalConv1x1WQuant_hpp

#import "backend/metal/MetalConvolutionCommon.hpp"

#if MNN_METAL_ENABLED && defined(MNN_LOW_MEMORY)

#include "MetalConv1x1Decode.hpp"
#include "MetalConv1x1Prefill.hpp"

namespace MNN {

// Device tiers, the profile ladder, the GEMV geometry helpers and the GEMV variant
// choice are all pure functions of the facts handed in and live in
// MetalConv1x1Decode.hpp. What remains here is only the adapter that reads Metal's
// device model and the class that builds and encodes the chosen plan.
Conv1x1DeviceProfile conv1x1DeviceProfile(const MetalGpuInfo& gpu);

class MetalConv1x1WQuant : public MetalConvolutionCommon {
public:
    MetalConv1x1WQuant(Backend* backend, const MNN::Op* op);
    virtual ~MetalConv1x1WQuant() = default;
    virtual MetalConv1x1WQuant* asWQuant() override { return this; }
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    void requestRowScale(const Tensor* scale) override { mRowScaleOffer = scale; }
    bool rowScaleFused() const override { return mPrepared && mFusedRowScale != nullptr; }
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    virtual void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                          id<MTLComputeCommandEncoder> encoder) override;
    // Also drops on geometry/stage changes and scratch re-creation, which onResize's pipeline check misses.
    virtual bool onReplayUpdate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    uint64_t replayStructure(const std::vector<Tensor*>& inputs) const;

    bool quantLayoutCompatible(const MetalConv1x1WQuant* peer) const;

    // Borrowed view of a prepared member. FusedLinear owns its own grouped
    // pipeline, grid and arguments; it never changes a member's dispatch.
    struct QuantMember {
        const Tensor* weight = nullptr;
        const Tensor* bias = nullptr;
        const Tensor* dequant = nullptr;
        id<MTLBuffer> params = nil;
        int bits = 0, blocks = 0, activation = 0;
        int inputChannels = 0, inputQuads = 0, outputChannels = 0, outputQuads = 0;
        float scaleCoef = 0.0f;
        explicit operator bool() const { return params != nil; }
    };
    QuantMember decodeMember() const;
    QuantMember widePrefillMember() const;
    id<MTLComputePipelineState> widePrefillPipeline() const;

    // Call before onResize; read prefillSiluMulFused() back afterwards.
    void requestPrefillSiluMul(const Tensor* gateOutput) { mPrefillSilu.gateOffer = gateOutput; }
    bool prefillSiluMulFused() const { return mPrepared && mPrefillPlan.fusion == PrefillFusion::SiluMul; }

    // Supersedes requestPrefillSiluMul when taken; same call-before-onResize contract.
    void requestPrefillGateUpDual(MetalConv1x1WQuant* gatePeer) { mPrefillSilu.dualPeerOffer = gatePeer; }
    bool prefillGateUpDualFused() const { return mPrepared && mPrefillPlan.fusion == PrefillFusion::GateUpDual; }
    float scaleCoef() const { return mScaleCoef; }

    std::shared_ptr<MNN::Tensor> getWeight() const { return mWeight; }
    std::shared_ptr<MNN::Tensor> getBias() const { return mBias; }
    std::shared_ptr<MNN::Tensor> getDequantScale() const { return mDequantScaleBias; }

#if MNN_METAL_OP_PROFILE
    const std::string& profileTag() const { return mProfileTag; }
    id<MTLComputePipelineState> selectedPipeline() const { return mMain.pipeline; }
    id<MTLComputePipelineState> selectedDequantPipeline() const { return mDequant.pipeline; }
    std::pair<MTLSize, MTLSize> dispatchThreads() const { return mMain.threads; }
    std::pair<MTLSize, MTLSize> dequantDispatchThreads() const { return mDequant.threads; }
    bool tensorApiFusedQuant() const { return mPrefillPlan.usesTensorApi() && mPrefillPlan.dequantInKernel(); }
    int gemmKPartitions() const { return mPrefillPlan.kPartitions; }

    struct RouteFacts {
        int dequantBits;
        int blockCount;
        int scratchGeneration;
        bool hasDequantMetadata;
        bool prepared;
        PrefillKernel kernel;
        WeightPreparation weightPreparation;
        bool usesTensorApi;
        PrefillFusion fusion;
        size_t tempWeightBytes, packedInputBytes, ksplitPartialBytes;
        id<MTLComputePipelineState> packPipeline;
        // Saved at prepare from the actual input; encode/Replay never recompute it.
        std::pair<MTLSize, MTLSize> packThreads;
        id<MTLComputePipelineState> ksplitReducePipeline;
        std::pair<MTLSize, MTLSize> ksplitReduceThreads;
        const char* profileName;
        int profileSource;
        int profileTgBudget;
    };
    RouteFacts routeFacts() const;
#endif

private:
    // One dispatch of the chosen prefill pipeline: the pipeline built from the routing's
    // description, plus the launch geometry recorded with it. Keeping the pair together
    // is the point - encode and Replay consume what prepare saved, and a new kernel
    // variant cannot install half of a pair the way eight flat fields allowed.
    struct PrefillStage {
        id<MTLComputePipelineState> pipeline = nil;
        std::pair<MTLSize, MTLSize> threads = {};
    };

    QuantMember quantMember() const;
    ErrorCode resizeImpl(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         bool allowTensorApi, QuantPrefillPlan& candidate);
    ErrorCode prepareDecodeConv(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    ErrorCode prepareQuantPrefill(const QuantPrefillPlan& plan, const QuantPrefillShape& shape,
                                  const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    ErrorCode prepareQuantScalar(const PrefillPipeline& stage, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs);
    // Build one stage from its description and record the launch geometry with it.
    // An explicit geometry overrides the one derived from the description, which is how
    // the input repack keeps its historical launch formula.
    ErrorCode buildPrefillStage(PrefillStage& stage, const PrefillPipeline& description, MetalBackend* backend,
                                const std::pair<MTLSize, MTLSize>* explicitThreads);
    // Reserve the scratch the description asks for, after every pipeline is known to
    // build. Only reservations acquired here are rolled back on failure.
    ErrorCode acquirePrefillScratch(const PrefillDescription& description, MetalBackend* backend);
    MetalConv1x1WQuant(Backend* backend, const MNN::Op* op, std::shared_ptr<MNN::Tensor> weight,
                       std::shared_ptr<MNN::Tensor> bias, std::shared_ptr<MNN::Tensor> dequantScale,
                       int dequantBits, float scaleCoef);
    QuantPrefillPlan mPrefillPlan;
    bool mPrepared = false;
    const Tensor* mRowScaleOffer = nullptr;
    const Tensor* mFusedRowScale = nullptr;

    // mMain is the convolution itself. The decode path installs its single dispatch there
    // too, which is why a prefill plan never clears it back to nil.
    PrefillStage mMain;
    PrefillStage mDequant;  // outer weight dequant, when the weights are not read in-kernel
    PrefillStage mPack;     // input repack for the device-tensor row path
    PrefillStage mReducer;  // split-K partial reduction
    // Scratch tensors, named by their content rather than by the stage that touches them
    // because two stages share each one: written by the stage on the left, read by mMain
    // (or by mReducer, for the split-K partials).
    std::shared_ptr<Tensor> mTempWeight;    // mDequant writes, mMain reads
    std::shared_ptr<Tensor> mPackedInput;   // mPack writes, mMain reads
    bool mDecodeFusionEligible = false;
    Conv1x1DeviceProfile mDeviceProfile = {};

    // Offers and dualCoef persist across resizes; the installed plan records the fold taken.
    struct PrefillSiluFold {
        const Tensor* gateOffer = nullptr;
        MetalConv1x1WQuant* dualPeerOffer = nullptr;
        id<MTLBuffer> dualCoef = nil;
    };
    PrefillSiluFold mPrefillSilu;

    int mBlockCount = 1;

    std::shared_ptr<Tensor> mKsplitPartial; // mMain writes, mReducer reads

    uint64_t mRecordedStructure = 0;
    // Bumped when a bound scratch Tensor is re-created, so a stale recording is
    // refused before validation can dereference the freed pointer.
    int mScratchGeneration = 0;

#if MNN_METAL_OP_PROFILE
    std::string mProfileTag;
#endif
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED && MNN_LOW_MEMORY */

#endif /* MetalConv1x1WQuant_hpp */