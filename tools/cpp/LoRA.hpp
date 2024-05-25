//
//  LoRA.hpp
//  MNN
//
//  Created by MNN on 2024/03/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LORA_HPP
#define LORA_HPP

#include "MNN_generated.h"

class LoRA {
public:
    LoRA(const char* originalModelFileName, const char* loraModelFileName);
    ~LoRA();
    void* getBuffer() const;
    const size_t getBufferSize() const;
    void apply_lora();
    void revert_lora();
private:
    std::unique_ptr<MNN::NetT> load_model(const char* name);
    void apply_external(MNN::OpT* conv, MNN::OpT* lora_A, MNN::OpT* lora_B);
private:
    LoRA();
    std::unique_ptr<MNN::NetT> mMNNNet, mLoRANet;
    std::unique_ptr<std::fstream> mExternalFile;
    void packMNNNet();
};

#endif // LORA_HPP
