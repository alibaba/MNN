//
//  CoreMLExecutor.h
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLDefine.h"
#include "CoreMLExecutorWrapper.h"
#include <memory>
#include <string>
#include <vector>
#include <MNN/Tensor.hpp>
#include "Model.pb-c.h"
@interface CoreMLExecutor : NSObject
#ifdef MNN_COREML_PREFILL_BENCH
- (bool)loadBenchmarkModel:(const std::string&)directory;
#endif
- (bool)buildModel:(CoreML__Specification__Model*)model weightsPath:(const std::string&)weightsPath
           weights:(const std::shared_ptr<std::vector<uint8_t>>&)weights forceFileCompilation:(bool)forceFileCompilation;
- (bool)invokeWithInputs:(const std::vector<std::pair<const MNN::Tensor*, std::string>>&)inputs
                 outputs:(const std::vector<std::pair<const MNN::Tensor*, std::string>>&)outputs flatIO:(bool)flatIO
                  images:(const MNN::CoreMLImageInputs&)images;
- (CoreMLExecutor*)sessionWithPrevious:(CoreMLExecutor*)previous;
@property int precision;
@property bool compiledFromSource;
@property(nonatomic, strong) MLModel* model;
@property(nonatomic, strong) MLState* state;
@end
