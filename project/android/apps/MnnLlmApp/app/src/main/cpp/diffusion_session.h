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
    explicit DiffusionSession(std::string  resource_path);
    void Run(const std::string& prompt, const std::string& image_path, const std::function<void(int)>& progressCallback);
private:
    std::string resource_path_;
    std::unique_ptr<Diffusion> diffusion_{nullptr};
};
}
