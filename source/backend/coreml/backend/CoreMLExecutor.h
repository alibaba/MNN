//
//  CoreMLExecutor.h
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLDefine.h"
#include <string>
#include <vector>
#include <MNN/Tensor.hpp>
#include "Model.pb-c.h"

struct View {
    int32_t offset = 0;
    int32_t stride[3] = {1, 1, 1};
};

struct Region {
    View src;
    View dst;
    int32_t size[3] = {1, 1, 1};
};


@interface CoreMLExecutor : NSObject

- (bool)invokeWithInputs:(const std::vector<std::pair<const MNN::Tensor*, std::string>>&)inputs
                 outputs:(const std::vector<std::pair<const MNN::Tensor*, std::string>>&)outputs API_AVAILABLE(ios(11));
- (NSURL*)saveModel:(CoreML__Specification__Model*)model API_AVAILABLE(ios(11));
- (bool)build:(NSURL*)modelUrl API_AVAILABLE(ios(11));
- (bool)cleanup;

@property MLModel* model API_AVAILABLE(ios(11));
@property NSString* mlModelFilePath;
@property NSString* compiledModelFilePath;
@property(nonatomic, readonly) int coreMlVersion;
@property __strong id<MLFeatureProvider> outputFeature API_AVAILABLE(ios(11));
@end

// RasterLayer
@interface RasterLayer : NSObject<MLCustomLayer> {
    struct SamplerInfo {
        unsigned int stride[4];     //stride[3] + offset
        unsigned int size[4];       //size[3] + totalSize
        unsigned int extent[4];     //dstStride[3]+dstOffset
        unsigned int imageSize[4];
    };
    std::vector<Region> regions;
    std::vector<SamplerInfo> samplers;
    std::vector<int> outputShape;
    id<MTLComputePipelineState> pipeline;
}
- (void)setRegionSampler;
- (std::pair<MTLSize, MTLSize>)computeBestGroupAndLocal:(SamplerInfo&) s;
- (MTLSize)computeBestGroup:(MTLSize)t;
@end

// DumpLayer
@interface DumpLayer : NSObject<MLCustomLayer>
@end
