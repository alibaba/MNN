//
// Created by ruoyi.sjd on 2024/01/12.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "diffusion_session.h"
#include "mls_log.h"
#include <memory>
#include <utility>
mls::DiffusionSession::DiffusionSession(std::string resource_path, int memory_mode):
                                        resource_path_(std::move(resource_path)),
                                        memory_mode_(memory_mode){
    this->diffusion_.reset(Diffusion::createDiffusion(
            resource_path_,
                          DiffusionModelType::STABLE_DIFFUSION_1_5,
                          MNNForwardType::MNN_FORWARD_OPENCL,
            memory_mode
            ));
    MNN_DEBUG("diffusion session init resource_path_: %s memory_mode: %d ", resource_path_.c_str(), memory_mode);
    // A failed load (missing text_encoder/unet/vae_decoder.mnn, or an OpenCL runtime that cannot
    // build the kernels) has to be reported: Run() would otherwise forward on null modules and
    // index an empty output vector, which is a SIGSEGV the Java layer cannot catch.
    loaded_ = this->diffusion_ && this->diffusion_->load();
    if (!loaded_) {
        MNN_ERROR("diffusion session load failed for %s", resource_path_.c_str());
    }
}

bool mls::DiffusionSession::Run(const std::string &prompt,
                                const std::string &image_path,
                                int iter_num,
                                int random_seed,
                                const std::function<void(int)>& progressCallback) {
    if (!loaded_) {
        loaded_ = this->diffusion_ && this->diffusion_->load();
    }
    if (!loaded_) {
        MNN_ERROR("diffusion session is not loaded, cannot run prompt: %s", prompt.c_str());
        return false;
    }
    bool success = this->diffusion_->run(prompt, image_path, iter_num, random_seed, progressCallback);
    if (memory_mode_ != 1) {
        loaded_ = false;
    }
    return success;
}
