#ifndef NNRSCENE_H
#define NNRSCENE_H
#include "NNRDefine.h"
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stdio.h>

typedef enum NNRRenderType {
    NNRRenderType_METAL = 0,
    NNRRenderType_OPENGL = 1,
    NNRRenderType_VULKAN = 2,
} NNRRenderType;

typedef enum NNRMapFlag {
    NNRMapFlag_READ = 1,
    NNRMapFlag_WRITE = 1 << 1,
} NNRMapFlag;

struct NNRContext;
struct NNRTarget;

typedef struct NNRTargetInfo {
    void* window;
    int width;
    int height;
    int viewport[4];
    int scissor[4];
    float depthRange[2];
    int clearDepth;
    int clearColor;
    float depthDefault;
    float backGroundColor[4];
    int colorFormat;
    int depthFormat;
    void* renderPassBeginInfo;// VkRenderPassBeginInfo*, Valid for vulkan
    uint8_t reserve[64];
} NNRTargetInfo;

NNR_PYTHON_FUNC struct NNRContext* NNRContextCreate(NNRRenderType type);
NNR_PYTHON_FUNC int NNRContextInit(struct NNRContext* target, void* user);
NNR_PYTHON_FUNC void NNRContextDestroy(struct NNRContext* target);

NNR_PYTHON_FUNC void NNRContextSetResourceDir(struct NNRContext* context, const char* dirName);

typedef enum NNRComputeMode {
    NNRComputeMode_ASYNC = 1,
    NNRComputeMode_PARREL = 1 << 1,
} NNRComputeMode;

NNR_PYTHON_FUNC int NNRContextSetComputeMode(struct NNRContext* context, int mask);

NNR_PYTHON_FUNC int NNRContextSetKey(struct NNRContext* context, const char* key, int value);

NNR_PYTHON_FUNC int NNRContextAddComponentsNeedDup(struct NNRContext* context, const char* key, int group);

NNR_PYTHON_FUNC int NNRContextSetComponentsDupNumber(struct NNRContext* context, int number, int group);

NNR_PYTHON_FUNC int NNRTargetInfoInit(struct NNRTargetInfo* targetInfo, int width, int height);

NNR_PYTHON_FUNC int NNRContextInitTarget(struct NNRContext* context, struct NNRTargetInfo* targetInfo);


NNR_PYTHON_FUNC struct NNRTarget* NNRTargetCreate(NNRRenderType type);
NNR_PYTHON_FUNC void NNRTargetDestroy(struct NNRTarget* target);

// If set pixels, after NNRSceneRender, it will read pixels to dst by described region
NNR_PYTHON_FUNC void NNRTargetSetPixels(struct NNRTarget* target, void* dst, int x, int y, int width, int height);


struct NNRScene;

NNR_PYTHON_FUNC struct NNRScene* NNRSceneCreateFromFile(const char* fileName ,struct NNRContext* context);
NNR_PYTHON_FUNC struct NNRScene* NNRSceneCreateFromBuffer(void* buffer, size_t length, struct NNRContext* context);
NNR_PYTHON_FUNC void NNRSceneDestroy(struct NNRScene* scene);


/**
 If filename or buffer is nullptr, it will delete the sub scene by key
 parent: nullptr means root scene
  flag: for future usage
 return value: 0: success, 1 error
 */
NNR_PYTHON_FUNC int NNRSceneReplaceFromFile(const char* fileName ,struct NNRScene* scene, const char* key, const char* parent, size_t flag);
NNR_PYTHON_FUNC int NNRSceneReplaceFromBuffer(void* buffer, size_t length, struct NNRScene* scene, const char* key, const char* parent, size_t flag);
NNR_PYTHON_FUNC int NNRSceneReset(struct NNRScene* scene, const char* key);


NNR_PYTHON_FUNC int NNRSceneSetComponentIndex(struct NNRScene* scene, int index, int group);

NNR_PYTHON_FUNC int NNRSceneRender(struct NNRScene* scene, struct NNRTarget* target);
NNR_PYTHON_FUNC int NNRSceneSetComponentData(struct NNRScene* scene, size_t pos, const void* ptr, size_t size);

NNR_PYTHON_FUNC int NNRSceneGetComponentSize(struct NNRScene* scene, size_t pos);
NNR_PYTHON_FUNC size_t NNRSceneGetComponentPosition(struct NNRScene* scene, const char* key);

/*Get Describe from Scene*/
NNR_PYTHON_FUNC const char* NNRSceneGetDescribe(struct NNRScene* scene);

enum {
    NNRTRACE_RECORD_TIME = 1 << 0,
    NNRTRACE_RECORD_MEMORY = 1 << 1,
};
NNR_PYTHON_FUNC void NNRSceneSetTraceConfig(struct NNRScene* scene, const char* traceDirectory, float samplerRate, uint64_t mask);

// For Internal C++
NNR_PUBLIC void* NNRSceneMapComponentData(struct NNRScene* scene, size_t pos, NNRMapFlag flag);
NNR_PUBLIC void NNRSceneUnmapComponentData(struct NNRScene* scene, size_t pos, void* ptr, NNRMapFlag flag);


// Return MNN::Tensor*
NNR_PUBLIC void* NNRSceneGetTensor(struct NNRScene* scene, size_t pos, NNRMapFlag flag);

NNR_PYTHON_FUNC size_t NNRSceneGetRenderMemoryHandle(struct NNRScene* scene, const char* key);

#ifdef __cplusplus
}
#endif

#endif
