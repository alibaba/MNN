//
// Created by ruoyi.sjd on 2024/01/12.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "diffusion_session.h"
#include "mls_log.h"
#include <memory>
#include <utility>
mls::DiffusionSession::DiffusionSession(std::string  resource_path): resource_path_(std::move(resource_path)) {
    this->diffusion_= std::make_unique<Diffusion>(
            resource_path_,
                          DiffusionModelType::STABLE_DIFFUSION_1_5,
                          MNNForwardType::MNN_FORWARD_OPENCL,
                          1,
                          20

            );
    MNN_DEBUG("diffusion session init resource_path_: %s ", resource_path_.c_str());
    this->diffusion_->load();
}

void mls::DiffusionSession::Run(const std::string &prompt, const std::string &image_path, const std::function<void(int)>& progressCallback) {
    this->diffusion_->run(prompt, image_path, progressCallback);
}
