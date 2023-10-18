//
//  llm.hpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//

#ifndef LLM_hpp
#define LLM_hpp

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <iostream>

#include <MNN/AutoTime.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include "tokenizer.hpp"

using namespace MNN;
using namespace Express;

class MNN_PUBLIC Llm {
public:
    Llm() {
        // default tokenier is senrencepiece
        tokenizer_.reset(new Sentencepiece);
    }
    static Llm* createLLM(const std::string& path);
    VARP gen_embedding(const std::vector<int>& input_ids);
    void load(const std::string& model_dir);
    int forward(const std::vector<int>& input_ids);
    std::vector<int> tokenizer_encode(const std::string& input_str);
    std::string decode(int id);
    std::string response(const std::string& input_str, std::ostream* os = &std::cout);
    float load_progress() { return load_progress_; }
    void reset();
private:
    virtual std::vector<int> tokenizer(const std::string& query) = 0;
    virtual VARP gen_attention_mask(int seq_len) = 0;
    virtual VARP gen_position_ids(int seq_len) = 0;
    virtual bool is_stop(int token_id) = 0;
protected:
    // model configs
    bool is_single_ = false;
    int layer_nums_ = 0;
    int hidden_size_ = 4096;
    std::vector<int> key_value_shape_ = {};
    std::string model_name_ = "";
    // gen info
    int gen_seq_len_ = 0;
    int all_seq_len_ = 0;
    int max_seq_len_ = 256;
    float load_progress_ = 0.f;
    // tokenizer
    std::unique_ptr<Tokenizer> tokenizer_;
private:
    // MNN Modules
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::vector<std::shared_ptr<Module>> modules_;
    std::vector<VARP> past_key_values_;
    // model dir
    std::string model_dir_;
    // tokenizer
    std::vector<std::string> word_decoder_;
    std::unordered_map<std::string, int> word_encoder_;
};

// some llm models
class Chatglm_6b : public Llm {
public:
    Chatglm_6b() {
        model_name_ = "Chatglm_6b";
        layer_nums_ = 28;
        key_value_shape_ = {2, 0, 1, 32, 128};
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
    virtual VARP gen_position_ids(int seq_len) override;
    virtual bool is_stop(int token_id) override;
    int context_len_ = 0;
};

class Chatglm2_6b : public Llm {
public:
    Chatglm2_6b() {
        model_name_ = "Chatglm2_6b";
        layer_nums_ = 28;
        key_value_shape_ = {2, 0, 1, 2, 128};
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
    virtual VARP gen_position_ids(int seq_len) override;
    virtual bool is_stop(int token_id) override;
};


class Qwen_7b : public Llm {
public:
    Qwen_7b() {
        model_name_ = "Qwen_7b";
        layer_nums_ = 32;
        key_value_shape_ = {2, 1, 0, 32, 128};
        tokenizer_.reset(new Tiktoken);
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
    virtual VARP gen_position_ids(int seq_len) override;
    virtual bool is_stop(int token_id) override;
};

class Llama2_7b : public Llm {
public:
    Llama2_7b() {
        model_name_ = "Llama2_7b";
        layer_nums_ = 32;
        key_value_shape_ = {2, 1, 32, 0, 128};
    }
private:
    virtual std::vector<int> tokenizer(const std::string& query) override;
    virtual VARP gen_attention_mask(int seq_len) override;
    virtual VARP gen_position_ids(int seq_len) override;
    virtual bool is_stop(int token_id) override;
};

#endif // LLM_hpp
