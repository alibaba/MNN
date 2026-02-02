
#ifndef MNN_DIFFUSION_SANA_LLM_hpp
#define MNN_DIFFUSION_SANA_LLM_hpp

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <algorithm>
#if defined(__APPLE__) && defined(__MACH__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
#include <MNN/llm/llm.hpp>
#else
#ifdef MNN_AAPL_FMWK
#include <MNN/llm/llm.hpp>
#else
#include "llm/llm.hpp"
#endif
#endif
#else
#ifdef MNN_AAPL_FMWK
#include <MNN/llm/llm.hpp>
#else
#include "llm/llm.hpp"
#endif
#endif
//#include "llm/llm.hpp"
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {
namespace DIFFUSION {

using namespace MNN::Express;
using namespace MNN::Transformer;

class SanaLlm {
public:
    SanaLlm(const std::string& config_path) {
        // Load LLM
        mLlm.reset(Llm::createLLM(config_path + "/config.json"));
        mLlm->set_config("{\"hidden_states\":true}");
        mLlm->load();
        
        // Load Meta Queries (Assuming meta_queries.mnn is in the same dir)
        std::string meta_path = config_path + "/meta_queries.mnn";
        mMetaQueries = Variable::load(meta_path.c_str());
        if (!mMetaQueries.empty()) {
            mMetaQueries[0].fix(VARP::CONSTANT);
        } else {
            MNN_ERROR("Failed to load meta_queries.mnn from %s\n", meta_path.c_str());
        }
    }

    // 处理单个或多个prompts（用于CFG）
    // use_cfg: 如果为true，则生成batch_size=2（用于CFG），negative_prompt可以为空字符串
    VARP process(const std::string& prompt, bool use_cfg = false, const std::string& negative_prompt = "") {
        AUTOTIME;
        if (mMetaQueries.empty()) {
            return nullptr;
        }

        // 1. Prepare Prompts
        std::string img_start_token = "<|vision_bos|>";
        std::string instruction_fmt = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n";
        std::string generator_fmt = "Generate an image: {input}";
        
        auto format_prompt = [&](const std::string& input) {
            std::string text = generator_fmt;
            std::string placeholder = "{input}";
            size_t pos = text.find(placeholder);
            if (pos != std::string::npos) {
                text.replace(pos, placeholder.length(), input);
            }
            std::string full = instruction_fmt;
            pos = full.find(placeholder);
            if (pos != std::string::npos) {
                full.replace(pos, placeholder.length(), text);
            }
            return full + img_start_token;
        };

        std::vector<std::string> prompts;
        // 如果use_cfg=true，则batch_size=2（用于CFG）
        if (use_cfg) {
            prompts.push_back(format_prompt(prompt));          // Positive prompt
            prompts.push_back(format_prompt(negative_prompt)); // Negative prompt（可以为空字符串）
        } else {
            prompts.push_back(format_prompt(prompt));
        }

        // 2. Tokenize & Pad
        std::vector<std::vector<int>> all_ids;
        int max_len = 0;
        for (const auto& p : prompts) {
            auto ids = mLlm->tokenizer_encode(p);
            all_ids.push_back(ids);
            if (ids.size() > max_len) max_len = ids.size();
        }

        int batch = all_ids.size(); // 2
        // Flatten input_ids and create attention_mask
        std::vector<int> flat_input_ids;
        std::vector<int> flat_mask;
        int pad_id = 151643; // Qwen default pad, but ideally should get from mLlm if possible. 
                             // Using hardcoded for now or try to finding eos.
                             // Actually mLlm->tokenizer_encode("") might give empty.
                             // Qwen eos is often 151643.
        
        // Right padding
        for (const auto& ids : all_ids) {
            int len = ids.size();
            int pad_len = max_len - len;
            for (int id : ids) {
                flat_input_ids.push_back(id);
                flat_mask.push_back(1);
            }
            for (int i = 0; i < pad_len; ++i) {
                flat_input_ids.push_back(pad_id); 
                flat_mask.push_back(0);
            }
        }

        auto input_ids_var = _Const(flat_input_ids.data(), {batch, max_len}, NCHW, halide_type_of<int>());
        // embedding expects input_ids
        auto inputs_embeds = mLlm->embedding(flat_input_ids); 
        // Reshape to [Batch, SeqLen, Hidden]
        inputs_embeds = _Reshape(inputs_embeds, {batch, max_len, -1});

        // 3. Meta Queries
        int num_queries = 256; 
        auto meta_queries = mMetaQueries[0]; // [256, H]
        // Expand meta_queries to [Batch, NumQueries, H]
        auto meta_queries_batch = _Unsqueeze(meta_queries, {0}); // [1, 256, H]
        
        // 如果batch > 1，需要复制meta_queries
        if (batch > 1) {
            std::vector<VARP> meta_list;
            for (int i = 0; i < batch; ++i) {
                meta_list.push_back(meta_queries_batch);
            }
            meta_queries_batch = _Concat(meta_list, 0); // [Batch, 256, H]
        }

        // 4. Concatenate
        auto full_embeds = _Concat({inputs_embeds, meta_queries_batch}, 1); // Concat on SeqLen dim (1)
        
        // 5. Update Attention Mask
        // Original mask [Batch, SeqLen]. we need [Batch, SeqLen + NumQueries]
        // The Llm->forwardRaw expects attention_mask as [Batch, 1, TotalLen, TotalLen] usually for causal, 
        // or [Batch, TotalLen] depending on impl.
        // It constructs `aug_attention_mask` of shape [Batch, SeqLen+NumQueries].
        
        int total_len = max_len + num_queries;
        
        // Qwen is Causal. But here 'verify_onnx.py' uses:
        // aug_attention_mask = cat([mask, ones(queries)])
        // position_ids = cumsum(aug_attention_mask) - 1
        // It passes `aug_attention_mask` to model. HuggingFace model handles 2D mask.
        
#if 1
        std::vector<int> mask_data(batch * total_len * total_len, 0);
        // For each batch:
        // 0..max_len are processing text.
        // max_len..total_len are queries.
        // verify_onnx.py uses `attention_mask` (boolean). HuggingFace converts this to causal mask usually?
        // Wait, verify_onnx: `input_ids` are text. `hidden_states` (meta queries) are appended.
        // It seems to run a single forward pass.
        // The mask in HF is `aug_attention_mask` which is 1s for valid text and 1s for queries. 0 for pad.
        // Since it's CausalLM, it will use causal mask automatically in HF.
        // So tokens can only attend to previous tokens.
        
        // Construct Causal Mask [Batch, 1, TotalLen, TotalLen]
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < total_len; ++i) {
                for (int j = 0; j < total_len; ++j) {
                    // Causal: j <= i
                    if (j > i) continue; // kept -inf

                    // But we must respect the padding of the TEXT part.
                    // Text part is indices 0..max_len-1.
                    // Queries are max_len..total_len-1.
                    
                    // Logic:
                    // If i < max_len: it's text token.
                    // If i >= max_len: it's query token.
                    
                    // The `flat_mask` tells us valid text tokens.
                    // flat_mask index for batch b, pos j (if j < max_len)
                    
                    bool valid_j = true;
                    if (j < max_len) {
                        if (flat_mask[b * max_len + j] == 0) valid_j = false;
                    }
                    // Queries (j >= max_len) are always valid.
                    
                    if (valid_j) {
                        mask_data[b * total_len * total_len + i * total_len + j] = 1;
                    }
                }
            }
        }
        auto attention_mask = _Const(mask_data.data(), {batch, 1, total_len, total_len}, NCHW, halide_type_of<int>());
#else
        std::vector<float> mask_data(batch * total_len * total_len, std::numeric_limits<float>::lowest());
        // For each batch:
        // 0..max_len are processing text.
        // max_len..total_len are queries.
        // verify_onnx.py uses `attention_mask` (boolean). HuggingFace converts this to causal mask usually?
        // Wait, verify_onnx: `input_ids` are text. `hidden_states` (meta queries) are appended.
        // It seems to run a single forward pass.
        // The mask in HF is `aug_attention_mask` which is 1s for valid text and 1s for queries. 0 for pad.
        // Since it's CausalLM, it will use causal mask automatically in HF.
        // So tokens can only attend to previous tokens.
        
        // Construct Causal Mask [Batch, 1, TotalLen, TotalLen]
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < total_len; ++i) {
                for (int j = 0; j < total_len; ++j) {
                    // Causal: j <= i
                    if (j > i) continue; // kept -inf

                    // But we must respect the padding of the TEXT part.
                    // Text part is indices 0..max_len-1.
                    // Queries are max_len..total_len-1.
                    
                    // Logic:
                    // If i < max_len: it's text token.
                    // If i >= max_len: it's query token.
                    
                    // The `flat_mask` tells us valid text tokens.
                    // flat_mask index for batch b, pos j (if j < max_len)
                    
                    bool valid_j = true;
                    if (j < max_len) {
                        if (flat_mask[b * max_len + j] == 0) valid_j = false;
                    }
                    // Queries (j >= max_len) are always valid.
                    
                    if (valid_j) {
                        mask_data[b * total_len * total_len + i * total_len + j] = 0.0f;
                    }
                }
            }
        }
        auto attention_mask = _Const(mask_data.data(), {batch, 1, total_len, total_len}, NCHW, halide_type_of<float>());
#endif
        
        // Position IDs
        // verify_onnx: position_ids = cumsum(aug_attention_mask) - 1.
        // aug_attention_mask is 1 for valid text and queries, 0 for pads.
        std::vector<int> pos_ids(batch * total_len);
        for (int b = 0; b < batch; ++b) {
            int cumsum = 0;
            for (int i = 0; i < total_len; ++i) {
                int val = 1; // query is 1
                if (i < max_len) {
                    val = flat_mask[b * max_len + i];
                }
                cumsum += val;
                int pos = cumsum - 1;
                if (pos < 0) pos = 0;
                pos_ids[b * total_len + i] = pos;
            }
        }
        auto position_ids = _Const(pos_ids.data(), {batch, total_len}, NCHW, halide_type_of<int>());

        // 6. Forward
        // Llm::forwardRaw typically returns vector<VARP> (logits, user fields...)
        // We enabled "output_hidden_states" in HF.

        mLlm->setKVCacheInfo(batch*total_len, 0);
        auto outputs = mLlm->forwardRaw(full_embeds, attention_mask, position_ids);
        int hiddenStateIndex = mLlm->getOutputIndex("hidden_states");
        hiddenStateIndex = hiddenStateIndex == -1 ? outputs.size() - 1 : hiddenStateIndex;

        // Get hidden_states ouptput
        auto output = outputs[hiddenStateIndex]; // Assume this is the correct output
        output = _Reshape(output, {batch, -1, output->getInfo()->dim[2]});

        // 7. Slice last 256 tokens
        // output matches [Batch, TotalLen, HiddenSize] (if it is hidden states)
        // We want [Batch, 256, HiddenSize], corresponding to the queries.
        // Queries are at the end. [TotalLen-NumQueries .. TotalLen]
        auto splits = _Split(output, {total_len - num_queries, num_queries}, 1); 
        // splits[0] is text part, splits[1] is query part.
        
#if FREE_LLM_INSTANCE
        ((MNN::Tensor*)(splits[1]->getTensor()))->wait(Tensor::MAP_TENSOR_READ, true);
        mLlm.reset();
#endif
        return splits[1];
    }

private:
    std::unique_ptr<Llm> mLlm;
    std::vector<VARP> mMetaQueries; // Store as VARP vector just in case
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_DIFFUSION_SANA_LLM_hpp
