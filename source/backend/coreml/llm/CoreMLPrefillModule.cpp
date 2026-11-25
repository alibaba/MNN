//
//  CoreMLPrefillModule.cpp
//  MNN
//
//  Created by MNN on 2026/11/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/coreml/backend/CoreMLPrefillModule.hpp"
#include "backend/coreml/backend/CoreMLTransformer.hpp"
#include "core/AttentionMaskConvention.hpp"
#include "core/KVMeta.hpp"
#include "core/KVCacheImport.hpp"
#include "llmconfig.hpp"
#include "MNN_generated.h"
#include <half.hpp>

namespace MNN {
namespace Transformer {
using namespace Express;
namespace {
class CoreMLPrefillModule : public Module {
public:
    std::shared_ptr<CoreMLTransformer> transformer;
    std::shared_ptr<Module> rotary, head;
    KVMeta* meta = nullptr;
    std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        if (inputs.size() != 4 || !meta || meta->previous != meta->remove || meta->n_reserve || meta->spec_block ||
            meta->file_flag != KVMeta::NoChange) {
            return {};
        }
        // The layer program has an ordinary causal mask. Reject other attention semantics.
        auto mask = inputs[1]->getInfo();
        if (!mask || !IsAttentionMaskCausalSentinel(true, mask->dim.size()) ||
            !transformer->supportsAttentionScale(meta->attn_scale)) {
            MNN_ERROR("[CoreML] complete-layer prefill requires the causal mask sentinel\n");
            return {};
        }
        auto hidden = inputs[0]->getInfo();
        if (!hidden || hidden->type != halide_type_of<float>() ||
            hidden->size != meta->add * transformer->hiddenSize() || !inputs[0]->readMap<float>()) {
            return {};
        }
        auto rot = rotary->onForward({inputs[2]});
        if (rot.size() != 2 || !rot[0]->readMap<float>() || !rot[1]->readMap<float>()) {
            return {};
        }
        // readMap synchronizes any Metal producer before Core ML consumes the host arrays.
        std::unique_ptr<Tensor> x(Tensor::create<float>(hidden->dim, const_cast<float*>(inputs[0]->readMap<float>())));
        std::unique_ptr<Tensor> cosine(
            Tensor::create<float>(rot[0]->getInfo()->dim, const_cast<float*>(rot[0]->readMap<float>())));
        std::unique_ptr<Tensor> sine(
            Tensor::create<float>(rot[1]->getInfo()->dim, const_cast<float*>(rot[1]->readMap<float>())));
        std::shared_ptr<KVCacheImport> pending;
        auto output = transformer->run(x.get(), cosine.get(), sine.get(), pending);
        if (!output) {
            return {};
        }
        auto data = _Input({output->length(0), transformer->hiddenSize(), 1, 1}, NCHW, halide_type_of<float>());
        auto ptr = data->writeMap<float>();
        auto src = reinterpret_cast<const half_float::half*>(output->host<void>());
        for (int i = 0; i < output->elementSize(); ++i) {
            ptr[i] = float(src[i]);
        }
        auto logits = head->onForward({_Convert(data, NC4HW4), inputs[3]});
        if (logits.size() != 1 || !logits[0]->readMap<float>()) {
            return {};
        }
        // Publish only after every layer and the native vocabulary head completed successfully.
        meta->pendingImport = std::move(pending);
        return logits;
    }
};
} // namespace
std::shared_ptr<Module> createCoreMLPrefillModule(const LlmConfig& config, const std::string& model,
                                                  const std::string& weights,
                                                  const std::shared_ptr<Executor::RuntimeManager>& runtime,
                                                  KVMeta* meta) {
    auto graph = config.resource(model);
    AutoStorage<uint8_t> storage;
    auto buffer = graph.data;
    size_t length = graph.size;
    if (!buffer) {
        FileLoader reader(graph.filePath().c_str(), true);
        if (!reader.valid() || !reader.read() || !reader.merge(storage)) {
            MNN_ERROR("[CoreML] cannot read graph %s\n", graph.filePath().c_str());
            return nullptr;
        }
        buffer = storage.get();
        length = storage.size();
    }
    flatbuffers::Verifier verify(buffer, length);
    if (!VerifyNetBuffer(verify)) {
        MNN_ERROR("[CoreML] invalid graph (%zu bytes)\n", length);
        return nullptr;
    }
    auto weight = config.resource(weights.empty() ? model + ".weight" : weights);
    if (!weight.valid() || weight.data) {
        MNN_ERROR("[CoreML] external weights require a file resource\n");
        return nullptr;
    }
    const auto format = config.config_.value("coreml_prefill_weights", "fp16");
    if (format != "fp16" && format != "int8_per_channel") {
        MNN_ERROR("[CoreML] unknown prefill weight format %s\n", format.c_str());
        return nullptr;
    }
    auto transformer = CoreMLTransformer::create(GetNet(buffer), weight.filePath(), format == "int8_per_channel");
    if (!transformer) {
        MNN_ERROR("[CoreML] model does not match the supported complete transformer-layer pattern\n");
        return nullptr;
    }
    std::shared_ptr<CoreMLPrefillModule> module(new CoreMLPrefillModule);
    module->transformer = transformer;
    module->meta = meta;
    Module::Config options;
    options.shapeMutable = true;
    options.rearrange = true;
    module->rotary.reset(
        config.loadModule({"position_ids"}, transformer->rotaryInputs(), model, runtime, &options, weights));
    module->head.reset(
        config.loadModule({transformer->headInput(), "logits_index"}, {"logits"}, model, runtime, &options, weights));
    if (!module->rotary || !module->head) {
        return nullptr;
    }
    module->registerModel({module->rotary, module->head});
    return module;
}
} // namespace Transformer
} // namespace MNN
