//
//  Options.hpp
//  MNN
//
//  Created by MNN on 2020/07/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_COMMON_OPTIONS_HPP_
#define MNN_CONVERTER_COMMON_OPTIONS_HPP_

#include <string>

#include "../source/compression/PipelineBuilder.hpp"

namespace common {

// TODO(): Refine
typedef struct Options {
    bool doCompress;
    compression::Pipeline compressionPipeline;
} Options;

Options DefaultOptions();

Options BuildOptions(const std::string& compressionFile);

}  // namespace common

#endif  // MNN_CONVERTER_COMMON_OPTIONS_HPP_
