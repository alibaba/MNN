//
//  CoreMLTransformer.hpp
//  MNN
//
//  Created by MNN on 2026/11/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREML_TRANSFORMER_HPP
#define MNN_COREML_TRANSFORMER_HPP
#include <MNN/Tensor.hpp>
#include <memory>
#include <string>
#include <vector>

namespace MNN {
struct Net;
struct KVCacheImport;
class MNN_PUBLIC CoreMLTransformer {
public:
    static std::shared_ptr<CoreMLTransformer> create(const Net* net, const std::string& weights);
    static std::shared_ptr<CoreMLTransformer> create(const Net* net, const std::string& weights, bool channelInt8);
    ~CoreMLTransformer();
    const std::vector<std::string>& rotaryInputs() const;
    const std::string& headInput() const;
    int hiddenSize() const;
    bool supportsAttentionScale(float scale) const;
    // Each invocation starts an independent causal prefix. Decode owns its native KV cache.
    std::shared_ptr<Tensor> run(const Tensor* hidden, const Tensor* cosine, const Tensor* sine,
                                std::shared_ptr<KVCacheImport>& state);

private:
    CoreMLTransformer();
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
} // namespace MNN
#endif
