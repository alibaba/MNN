//
// Created by ruoyi.sjd on 2024/01/12.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once
#include <string>
#include "diffusion/diffusion.hpp"

using namespace MNN::DIFFUSION;
namespace mls {
class DiffusionSession {
public:
    explicit DiffusionSession(std::string  resource_path, int memory_mode);
    void Run(const std::string& prompt, const std::string& image_path,
             int iter_num,
             int random_seed, const std::function<void(int)>& progressCallback);
private:
    bool loaded_{false};
    std::string resource_path_;
    int memory_mode_;
    std::unique_ptr<Diffusion> diffusion_{nullptr};
};
}
