#ifndef NNROpenGL_h
#define NNROpenGL_h
#include "NNRDefine.h"
#ifdef __cplusplus
extern "C" {
#endif

NNR_PYTHON_FUNC void NNRContextSetVersion(struct NNRContext* dst, int majorVersion, int minorVersion);
NNR_PYTHON_FUNC void NNRTargetSetViewport(struct NNRTarget* dst, int x, int y, int w, int h);
NNR_PYTHON_FUNC void NNRTargetSetClearBit(struct NNRTarget* dst, uint32_t mask);
NNR_PYTHON_FUNC void NNRTargetSetDepthValue(struct NNRTarget* dst, float value);
NNR_PYTHON_FUNC void NNRTargetSetBackgroundColor(struct NNRTarget* dst, float r, float g, float b, float a);

NNR_PYTHON_FUNC void NNRTargetSetExtraTextureInfo(struct NNRTarget* dst, int compIndex, uint32_t textureId, uint32_t target, uint32_t width, uint32_t height, uint32_t depth);

#ifdef __cplusplus
}
#endif

#endif
