#ifndef NNRUtils_h
#define NNRUtils_h
#include "NNRDefine.h"
#ifdef __cplusplus
extern "C" {
#endif

// helper function for offline render, only support 4 channels of image
NNR_PYTHON_FUNC void NNRHeadlessRendering(struct NNRScene* scene, struct NNRTarget* target, void* imageSrc, int width, int height);

#ifdef __cplusplus
}
#endif

#endif