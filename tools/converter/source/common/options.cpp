//
//  options.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "options.hpp"

namespace common {

Options DefaultOptions() {
    Options opt;
    opt.doCompress = false;
    opt.compressionPipeline = compression::Pipeline();
    return opt;
}

Options BuildOptions(const std::string& compressionFile) {
    Options options;
    if (!compressionFile.empty()) {
        options.doCompress = true;
    } else {
        options.doCompress = false;
    }
    // TODO(): Check if file exist.
    compression::PipelineBuilder builder(compressionFile);
    options.compressionPipeline = builder.Build();
    return std::move(options);
}

}  // namespace common
