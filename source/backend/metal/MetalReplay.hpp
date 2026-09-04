//
//  MetalReplay.hpp
//  MNN
//
//  Encode replay ("MNN-level indirect command buffer") for stable-shape decode
//  forwards. A MetalExecution whose per-token encode is identical (same buffer
//  bindings) is captured once via an NSProxy standing in for the compute
//  encoder; later tokens re-emit the captured call list directly, skipping the
//  op's onEncode logic (pipeline selection, branchy dispatch code, ARC churn).
//
//  Safety model: every tensor-backed binding is re-validated against the
//  tensor's CURRENT device buffer+offset before each replay. Any mismatch
//  (allocator reshuffle, KV-cache expansion, shape change) drops the recording
//  and falls back to the normal onEncode path, which may then re-record.
//
//  LIFETIME INVARIANT: that revalidation dereferences the recorded raw Tensor*,
//  so a Tensor annotated into a recording MUST OUTLIVE the recording. Freeing
//  one is a heap-use-after-free inside metalReplayValidate, not a mismatch it
//  can catch. An execution that owns a Tensor it binds must therefore either
//    - keep the object for its lifetime and only resize it in place
//      (MetalAttention::mTempQK, MetalConvolutionWinograd::mTempSrc), or
//    - bump a generation counter when it really re-allocates and refuse replay
//      from onReplayUpdate while it differs (MetalLinearAttention::mConvOut).
//  Note _replayHashIO covers only the op's own inputs/outputs, so it does not
//  protect internally-owned tensors. A parent op that drives child executions
//  (MetalFusedProj) captures the children's bindings into ITS recording, so the
//  parent is the one that must guard against a child's tensor churn.
//
//  Ops whose encode depends on per-token CPU state (attention: kv-length in
//  grids and param-buffer contents) must override canRecordEncode() -> false
//  or implement onReplayUpdate().
//

#ifndef MetalReplay_hpp
#define MetalReplay_hpp

#include "MetalDefine.h"
#include <vector>
#include <cstdint>
#include <utility>

#if MNN_METAL_ENABLED
namespace MNN {
class Tensor;

struct MetalReplayBinding {
    int index = 0;
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
    // Non-null when the binding came from MetalBackend::setTensor: replay
    // validates the tensor's current buffer+offset against the recorded one.
    const Tensor* tensor = nullptr;
};

struct MetalReplayEvent {
    enum Type : uint8_t { Dispatch, DispatchThreads, BarrierScope } type = Dispatch;
    id<MTLComputePipelineState> pipeline = nil;
    std::vector<MetalReplayBinding> bindings;
    std::vector<std::pair<int, std::vector<uint8_t>>> bytesArgs;
    std::vector<std::pair<NSUInteger, NSUInteger>> tgMemory; // (index, length)
    MTLSize grid{0, 0, 0};
    MTLSize threads{0, 0, 0};
    NSUInteger barrierScope = 0; // BarrierScope events only
};

// Validate all tensor-backed bindings against their current device buffer
// + offset. Exposed so onReplayUpdate hooks that mutate per-token state can
// check staleness BEFORE committing to the replay (a late metalReplayEmit
// failure would fall back to onEncode and apply the state mutation twice).
bool metalReplayValidate(const std::vector<MetalReplayEvent>& events);

// Validate all tensor-backed bindings, then re-emit the recorded calls onto
// the encoder. Returns false (nothing emitted) if any binding is stale.
bool metalReplayEmit(const std::vector<MetalReplayEvent>& events, id<MTLComputeCommandEncoder> encoder);

} // namespace MNN

#ifdef __OBJC__
// Stands in for id<MTLComputeCommandEncoder> during a recording encode. Every
// message is forwarded to the real encoder (so the recording token executes
// normally); the replayable subset is additionally captured into `events`.
// Any unsupported selector marks the recording failed and the op is banned
// from replay (permanent normal path).
@interface MetalReplayProxy : NSProxy
@property(nonatomic, readonly) BOOL failed;
- (instancetype)initWithTarget:(id<MTLComputeCommandEncoder>)target
                        events:(std::vector<MNN::MetalReplayEvent>*)events;
- (void)annotateTensor:(const MNN::Tensor*)tensor atIndex:(int)index;
@end

// Set by MetalExecution::onExecute only while a recording encode is in flight;
// lets MetalBackend::setTensor annotate the binding it just recorded with the
// source tensor for replay-time validation. Nil outside recording.
// thread_local: concurrent backends on different threads must not see each
// other's recording proxy (cross-annotation + ARC strong-store race -> crash).
// __unsafe_unretained is safe: the proxy is strongly held on the recording
// thread's stack (MetalExecution::onExecute) for the whole window.
extern thread_local __unsafe_unretained MetalReplayProxy* gMetalReplayProxy;
#endif

#endif /* MNN_METAL_ENABLED */
#endif /* MetalReplay_hpp */
