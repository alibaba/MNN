//
//  MetalReplay.mm
//  MNN
//
//  Created by MNN. Recording proxy + replay emitter for the encode-replay path.
//

#import "MetalReplay.hpp"
#import "MetalBackend.hpp"
#include "core/TensorUtils.hpp"

#if MNN_METAL_ENABLED

#ifdef __OBJC__
thread_local __unsafe_unretained MetalReplayProxy* gMetalReplayProxy = nil;

@implementation MetalReplayProxy {
    id<MTLComputeCommandEncoder> _target;
    std::vector<MNN::MetalReplayEvent>* _events;
    MNN::MetalReplayEvent _pending;
    BOOL _hasPipeline;
}
@synthesize failed = _failed;

- (instancetype)initWithTarget:(id<MTLComputeCommandEncoder>)target
                        events:(std::vector<MNN::MetalReplayEvent>*)events {
    _target = target;
    _events = events;
    _hasPipeline = NO;
    _failed = NO;
    return self;
}

- (NSMethodSignature*)methodSignatureForSelector:(SEL)sel {
    return [(NSObject*)_target methodSignatureForSelector:sel];
}

- (void)forwardInvocation:(NSInvocation*)inv {
    // Unrecorded selector: execute normally, poison the recording.
    _failed = YES;
    [inv invokeWithTarget:_target];
}

- (void)_resetPendingAfterDispatch {
    _pending.bindings.clear();
    _pending.bytesArgs.clear();
    _pending.tgMemory.clear();
    // Pipeline state persists across dispatches in a real encoder; keep it.
}

- (void)setComputePipelineState:(id<MTLComputePipelineState>)state {
    [_target setComputePipelineState:state];
    _pending.pipeline = state;
    _hasPipeline = YES;
}

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    [_target setBuffer:buffer offset:offset atIndex:index];
    MNN::MetalReplayBinding b;
    b.index = (int)index;
    b.buffer = buffer;
    b.offset = offset;
    b.tensor = nullptr;
    for (auto& cur : _pending.bindings) {
        if (cur.index == b.index) {
            cur = b;
            return;
        }
    }
    _pending.bindings.emplace_back(std::move(b));
}

- (void)setBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index {
    [_target setBufferOffset:offset atIndex:index];
    for (auto& cur : _pending.bindings) {
        if (cur.index == (int)index) {
            cur.offset = offset;
            return;
        }
    }
    _failed = YES; // offset update without a recorded binding: can't replay
}

- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    [_target setBytes:bytes length:length atIndex:index];
    std::vector<uint8_t> copy((const uint8_t*)bytes, (const uint8_t*)bytes + length);
    for (auto& cur : _pending.bytesArgs) {
        if (cur.first == (int)index) {
            cur.second = std::move(copy);
            return;
        }
    }
    _pending.bytesArgs.emplace_back((int)index, std::move(copy));
}

- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index {
    [_target setThreadgroupMemoryLength:length atIndex:index];
    _pending.tgMemory.emplace_back(index, length);
}

- (void)memoryBarrierWithScope:(MTLBarrierScope)scope {
    if ([_target respondsToSelector:@selector(memoryBarrierWithScope:)]) {
        [_target memoryBarrierWithScope:scope];
    }
    if (!_pending.bindings.empty() || _hasPipeline) {
        _failed = YES; // barrier mid-state: ordering we don't model
        return;
    }
    MNN::MetalReplayEvent e;
    e.type = MNN::MetalReplayEvent::BarrierScope;
    e.barrierScope = scope;
    _events->emplace_back(std::move(e));
}

- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage {
    // Resource usage hints are not captured; without them replay may be
    // incorrect for heap/argument-buffer resources, so ban this op.
    _failed = YES;
}

- (void)useHeap:(id<MTLHeap>)heap {
    _failed = YES;
}

- (void)dispatchThreadgroups:(MTLSize)threadgroups threadsPerThreadgroup:(MTLSize)threads {
    [_target dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
    if (!_hasPipeline || nil == _pending.pipeline) {
        _failed = YES;
        return;
    }
    _pending.type = MNN::MetalReplayEvent::Dispatch;
    _pending.grid = threadgroups;
    _pending.threads = threads;
    _events->push_back(_pending);
    [self _resetPendingAfterDispatch];
}

- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)threadsPerThreadgroup {
    [_target dispatchThreads:threads threadsPerThreadgroup:threadsPerThreadgroup];
    if (!_hasPipeline || nil == _pending.pipeline) {
        _failed = YES;
        return;
    }
    _pending.type = MNN::MetalReplayEvent::DispatchThreads;
    _pending.grid = threads;
    _pending.threads = threadsPerThreadgroup;
    _events->push_back(_pending);
    [self _resetPendingAfterDispatch];
}

- (void)annotateTensor:(const MNN::Tensor*)tensor atIndex:(int)index {
    // The matching setBuffer: was recorded just before this call.
    for (auto& cur : _pending.bindings) {
        if (cur.index == index && cur.tensor == nullptr) {
            cur.tensor = tensor;
            return;
        }
    }
}

@end
#endif // __OBJC__

namespace MNN {

bool metalReplayValidate(const std::vector<MetalReplayEvent>& events) {
    // Every tensor-backed binding is checked against its CURRENT address.
    for (const auto& e : events) {
        if (e.type == MetalReplayEvent::BarrierScope) {
            continue;
        }
        for (const auto& b : e.bindings) {
            if (b.tensor != nullptr) {
                if (b.tensor->deviceId() == 0) {
                    return false;
                }
                auto cur = MetalBackend::getBuffer(b.tensor);
                if (cur.first != b.buffer || (NSUInteger)cur.second != b.offset) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool metalReplayEmit(const std::vector<MetalReplayEvent>& events, id<MTLComputeCommandEncoder> encoder) {
    // Validate everything before emitting anything (a partial emit followed by
    // a normal encode would double-dispatch).
    if (!metalReplayValidate(events)) {
        return false;
    }
    id<MTLComputePipelineState> lastPipeline = nil;
    for (const auto& e : events) {
        if (e.type == MetalReplayEvent::BarrierScope) {
            if ([encoder respondsToSelector:@selector(memoryBarrierWithScope:)]) {
                [encoder memoryBarrierWithScope:(MTLBarrierScope)e.barrierScope];
            }
            continue;
        }
        if (e.pipeline != lastPipeline) {
            [encoder setComputePipelineState:e.pipeline];
            lastPipeline = e.pipeline;
        }
        for (const auto& b : e.bindings) {
            [encoder setBuffer:b.buffer offset:b.offset atIndex:b.index];
        }
        for (const auto& by : e.bytesArgs) {
            [encoder setBytes:by.second.data() length:by.second.size() atIndex:by.first];
        }
        for (const auto& tg : e.tgMemory) {
            [encoder setThreadgroupMemoryLength:tg.second atIndex:tg.first];
        }
        if (e.type == MetalReplayEvent::Dispatch) {
            [encoder dispatchThreadgroups:e.grid threadsPerThreadgroup:e.threads];
        } else {
            [encoder dispatchThreads:e.grid threadsPerThreadgroup:e.threads];
        }
    }
    return true;
}

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
