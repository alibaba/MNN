#ifndef NNRMetal_h
#define NNRMetal_h
#include "NNRDefine.h"
#import <Metal/Metal.h>
struct NNRContextMetalInfo {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    MTLPixelFormat colorPixelFormat;
};
struct NNRTargetMetalInfo {
    MTLRenderPassDescriptor* pass;
    MTLViewport viewport;
    id<MTLDrawable> currentDrawable;
};

#ifdef __cplusplus
extern "C" {
#endif

NNR_PUBLIC void NNRMetalContextSetInfo(struct NNRContext* dst, struct NNRContextMetalInfo* info);
NNR_PUBLIC void NNRMetalContextSetAmpCount(struct NNRContext* dst, int ampCount);

NNR_PUBLIC void NNRMetalTargetSetInfo(struct NNRTarget* dst, struct NNRTargetMetalInfo* info);
NNR_PUBLIC void NNRMetalTargetSetMultiViewport(struct NNRTarget* dst, MTLViewport* viewports, int viewportNumber);
NNR_PUBLIC void NNRMetalTargetSetCmdInfo(struct NNRTarget* dst, id<MTLCommandBuffer> buffer);

NNR_PUBLIC void NNRMetalSetTexture(size_t handle, id<MTLTexture> texture);


#ifdef __cplusplus
}
#endif

#endif
