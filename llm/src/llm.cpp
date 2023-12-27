//
//  llm.cpp
//
//  Created by MNN on 2023/08/25.
//  ZhaodeWang
//
// #define MNN_OPEN_TIME_TRACE 1

#include <iostream>

#include "llm.hpp"
#include "tokenizer.hpp"
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/AutoTime.hpp>

#include <fstream>

Llm* Llm::createLLM(const std::string& path, std::string model_type) {
    auto size = path.size();
    // end with '.mnn' is single model file, otherwise split block models
    bool is_single = (size > 4 &&
                      path[size - 4] == '.' &&
                      path[size - 3] == 'm' &&
                      path[size - 2] == 'n' &&
                      path[size - 1] == 'n');
    Llm* llm = nullptr;
    if (model_type == "auto") {
        model_type = path;
    }
    if (model_type.find("chatglm") != std::string::npos) {
        if (model_type.find("chatglm2") != std::string::npos) {
            llm = new Chatglm2_6b;
        } else if (model_type.find("chatglm3") != std::string::npos) {
            llm = new Chatglm2_6b;
            llm->model_name_ = "Chatglm3_6b";
        } else {
            llm = new Chatglm_6b;
        }
    } else if (model_type.find("codegeex2") != std::string::npos) {
        llm = new Chatglm2_6b;
        llm->model_name_ = "Codegeex2_6b";
    } else if (model_type.find("qwen") != std::string::npos) {
        if (model_type.find("1.8") != std::string::npos) {
            llm = new Qwen_1_8b;
        } else {
            llm = new Qwen_7b;
        }
    } else if (model_type.find("llama2") != std::string::npos) {
        llm = new Llama2_7b;
    } else if (model_type.find("baichuan") != std::string::npos) {
        llm = new Llama2_7b;
        llm->model_name_ = "Baichuan2_7b";
    }
    if (!llm) {
        std::cerr << "model type can't judge!" << std::endl;
        return llm;
    }
    llm->is_single_ = is_single;
    std::cout << "### model name : "<< llm->model_name_ << std::endl;
    return llm;
}

void Llm::chat() {
    while (true) {
        std::cout << "\nQ: ";
        std::string input_str;
        std::cin >> input_str;
        if (input_str == "/exit") {
            break;
        }
        if (input_str == "/reset") {
            reset();
            std::cout << "\nA: reset done." << std::endl;
            continue;
        }
        std::cout << "\nA: " << std::flush;
        response(input_str);
        std::cout << std::endl;
    }
    reset();
}

std::string Llm::response(const std::string& query, std::ostream* os, const char* end_with) {
    if (!end_with) {
        end_with = "\n";
    }
    // init status
    gen_seq_len_ = 0;
    all_seq_len_ = 0;
    prefill_us_ = 0;
    decode_us_ = 0;
    past_key_values_.clear();
    if (is_single_) {
        past_key_values_.push_back(_Input(key_value_shape_, NCHW));
    } else {
        for (int i = 0; i < layer_nums_; i++) {
            past_key_values_.push_back(_Input(key_value_shape_, NCHW));
        }
    }
    // response
    auto input_ids = tokenizer(query);
    if (!history_.empty()) {
        std::copy(input_ids.begin(), input_ids.end(), std::back_inserter(history_));
        input_ids = history_;
    } else {
        history_ = input_ids;
    }

    prompt_len_ = static_cast<int>(input_ids.size());
    auto st = std::chrono::system_clock::now();
    int token = forward(input_ids);
    auto et = std::chrono::system_clock::now();
    history_.push_back(token);
    std::string output_str = decode(token);
    prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    *os << output_str << std::flush;
    while (gen_seq_len_ < max_seq_len_) {
        st = std::chrono::system_clock::now();
        token = forward({token});
        et = std::chrono::system_clock::now();
        decode_us_ += std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        if (is_stop(token)) {
            *os << end_with << std::flush;
            break;
        }
        history_.push_back(token);
        auto word = decode(token);
        *os << word << std::flush;
        output_str += word;
    }
#ifdef DUMP_PROFILE_INFO
    print_speed();
#endif
    // update Cache
    // runtime_manager_->updateCache();
    // reset forward info
    return output_str;
}

void Llm::print_speed() {
    auto prefill_s = prefill_us_ * 1e-6;
    auto decode_s = decode_us_ * 1e-6;
    auto total_s = prefill_s + decode_s;
    printf("\n#################################\n");
    printf(" total tokens num  = %d\n", prompt_len_ + gen_seq_len_);
    printf("prompt tokens num  = %d\n", prompt_len_);
    printf("output tokens num  = %d\n", gen_seq_len_);
    printf("  total time = %.2f s\n", total_s);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("  total speed = %.2f tok/s\n", (prompt_len_ + gen_seq_len_) / total_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len_ / prefill_s);
    printf(" decode speed = %.2f tok/s\n", gen_seq_len_ / decode_s);
    printf("   chat speed = %.2f tok/s\n", gen_seq_len_ / total_s);
    printf("##################################\n");
}

void Llm::reset() {
    history_.clear();
}

void Llm::load(const std::string& model_dir) {
    model_dir_ = model_dir;
    // init
    ScheduleConfig config;
    BackendConfig cpuBackendConfig;
    config.type          = MNN_FORWARD_CPU;
    // config.type          = MNN_FORWARD_OPENCL;
    config.numThread     = 4;
    cpuBackendConfig.precision = BackendConfig::Precision_Low;
    cpuBackendConfig.memory = BackendConfig::Memory_Low;
    config.backendConfig = &cpuBackendConfig;
    runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
    if (config.type == MNN_FORWARD_OPENCL) {
        const char* cacheFileName = ".tempcache";
        // runtime_manager_->setCache(cacheFileName);
    }
    load_progress_ = 0.f;
    printf("load tokenizer\n");
    // 1. load vocab
    std::string tokenizer_path = model_dir + "/tokenizer.txt";
    load_progress_ += 5.f;
    tokenizer_->load(tokenizer_path);
    load_progress_ += 5.f;
    printf("load tokenizer Done\n");
    // 2. load model
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange = true;
    if (is_single_) {
        key_value_shape_.insert(key_value_shape_.begin(), layer_nums_);
        modules_.resize(1);
        std::string model_path = model_dir;
        std::string external_path = model_dir + ".weight";
        MNN_PRINT("load %s ... ", model_path.c_str());
        runtime_manager_->setExternalFile(external_path);
        modules_[0].reset(Module::load(
                {"input_ids", "attention_mask", "position_ids", "past_key_values"},
                {"token_id", "presents"}, model_path.c_str(), runtime_manager_, &module_config));
        MNN_PRINT("Done!\n");
        load_progress_ += 90.f;
    } else {
        // 2. load models
        modules_.resize(layer_nums_ + 2);
        float step = 90.0 / modules_.size();
        char buffer[50];
        // load lm model
        std::string lm_model_path = model_dir + "/lm.mnn";
        std::string embedding_model_path = model_dir + "/embedding.mnn";
        MNN_PRINT("[%3.0f%% ] load %s model ... ", load_progress_, lm_model_path.c_str());
        modules_[layer_nums_].reset(Module::load({}, {}, lm_model_path.c_str(), runtime_manager_, &module_config));
        MNN_PRINT("Done!\n");
        load_progress_ += step;
#ifndef USING_DISK_EMBED
        MNN_PRINT("[%3.0f%% ] load %s model ... ", load_progress_, embedding_model_path.c_str());fflush(stdout);
        modules_[layer_nums_ + 1].reset(Module::load({}, {}, embedding_model_path.c_str(), runtime_manager_, &module_config));
        MNN_PRINT("Done!\n");
        load_progress_ += step;
#endif
        // load glm_block models
        for (int i = 0; i < layer_nums_; i++) {
            load_progress_ += step;
            std::string model_path = model_dir + "/block_" + std::to_string(i) + ".mnn";
            MNN_PRINT("[%3.0f%% ] load %s model ... ", load_progress_, model_path.c_str());
            modules_[i].reset(Module::load(
                {"inputs_embeds", "attention_mask", "position_ids", "past_key_values"},
                {"hidden_states", "presents"}, model_path.c_str(), runtime_manager_, &module_config));
            MNN_PRINT("Done!\n");
        }
    }
    if (config.type == MNN_FORWARD_OPENCL) {
        // warmup();
    }
}

void Llm::warmup() {
    // warmup
    MNN_PRINT("### warmup ... ");
    if (is_single_) {
        past_key_values_.push_back(_Input(key_value_shape_, NCHW));
    } else {
        for (int i = 0; i < layer_nums_; i++) {
            past_key_values_.push_back(_Input(key_value_shape_, NCHW));
        }
    }
    std::vector<int> tmp(1, 0);
    forward(tmp);
    all_seq_len_ = 0;
    gen_seq_len_ = 0;
    printf("Done\n");
}

int Llm::forward(const std::vector<int>& input_ids) {
    int seq_len = input_ids.size();
    auto inputs_ids_ = _Const(input_ids.data(), {seq_len}, NCHW, halide_type_of<int>());
    auto attention_mask = gen_attention_mask(seq_len);
    auto position_ids = gen_position_ids(seq_len);
    int id = -1;
    if (is_single_) {
        // single model
        auto outputs = modules_.back()->onForward({inputs_ids_, attention_mask, position_ids, past_key_values_[0]});
        id = outputs[0]->readMap<int>()[0];
        past_key_values_[0] = outputs[1];
    } else {
        // split block models
#ifdef USING_DISK_EMBED
        auto hidden_states = disk_embedding(input_ids);
#else
        auto hidden_states = modules_[layer_nums_ + 1]->onForward({inputs_ids_})[0];
#endif
        for (int i = 0; i < layer_nums_; i++) {
            AUTOTIME;
            auto outputs = modules_[i]->onForward({hidden_states, attention_mask, position_ids, past_key_values_[i]});
            hidden_states = outputs[0];
            past_key_values_[i] = outputs[1];
        }
        {
            AUTOTIME;
            auto outputs = modules_[layer_nums_]->onForward({hidden_states});
            id = outputs[0]->readMap<int>()[0];
        }

    }
    all_seq_len_ += seq_len;
    gen_seq_len_++;
    return id;
}

VARP Llm::disk_embedding(const std::vector<int>& input_ids) {
    AUTOTIME;
    // disk embedding save memory
    size_t seq_len = input_ids.size();
    auto embedding = _Input({static_cast<int>(seq_len), 1, hidden_size_}, NCHW);
    size_t size = hidden_size_ * sizeof(int16_t);
    std::string file_path = model_dir_ + "/embeddings_bf16.bin";
    FILE* file = fopen(file_path.c_str(), "rb");
    std::unique_ptr<int16_t[]> buffer(new int16_t[hidden_size_]);
    for (size_t i = 0; i < seq_len; i++) {
        fseek(file, input_ids[i] * size, SEEK_SET);
        fread(buffer.get(), 1, size, file);
        auto ptr = embedding->writeMap<int16_t>() + i * hidden_size_ * 2;
        for (int j = 0; j < hidden_size_; j++) {
            ptr[j * 2] = 0;
            ptr[j * 2 + 1] = buffer[j];
        }
    }
    fclose(file);
    return embedding;
}

std::vector<int> Llm::tokenizer_encode(const std::string& input_str) {
    auto ids = tokenizer_->encode(input_str);
    return ids;
}

std::string Llm::decode(int id) {
    std::string word = tokenizer_->decode(id);
    // Fix utf-8 garbled characters
    if (word.length() == 6 && word[0] == '<' && word[word.length()-1] == '>' && word[1] == '0' && word[2] == 'x') {
        int num = std::stoi(word.substr(3, 2), nullptr, 16);
        word = static_cast<char>(num);
    }
    return word;
}

// Chatglm_6b
std::vector<int> Chatglm_6b::tokenizer(const std::string& query) {
    auto ids = tokenizer_encode(query);
    context_len_ = ids.size();
    ids.push_back(130001);
    ids.push_back(130004);
    return ids;
}

VARP Chatglm_6b::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = attention_mask->writeMap<int>();
    for (int i = 0; i < seq_len * seq_len; i++) {
        ptr[i] = 0;
    }
    if (seq_len > 1) {
        for (int i = 1; i < seq_len; i++) {
            ptr[seq_len * i - 1] = 1;
        }
    }
    return attention_mask;
}

VARP Chatglm_6b::gen_position_ids(int seq_len) {
    auto position_ids = _Input({1, 2, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = 1;
        ptr[1] = all_seq_len_ - context_len_;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i;
            ptr[seq_len + i] = 0;
        }
        ptr[2 * seq_len - 1] = 1;
    }
    return position_ids;
}

bool Chatglm_6b::is_stop(int token_id) {
    return token_id == 130005;
}

// Chatglm2_6b
std::vector<int> Chatglm2_6b::tokenizer(const std::string& query) {
    auto prompt = "问：" + query + "\n答：";
    auto ids = tokenizer_encode(prompt);
    if (history_.empty()) {
        ids.insert(ids.begin(), 64792);
        ids.insert(ids.begin(), 64790);
    }
    return ids;
}

VARP Chatglm2_6b::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = attention_mask->writeMap<int>();
    if (seq_len > 1) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                ptr[seq_len * i + j] = j > i;
            }
        }
    } else {
        ptr[0] = 0;
    }
    return attention_mask;
}

VARP Chatglm2_6b::gen_position_ids(int seq_len) {
    auto position_ids = _Input({seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = gen_seq_len_;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i;
        }
    }
    return position_ids;
}

bool Chatglm2_6b::is_stop(int token_id) {
    return token_id <= 2;
}

// Qwen_7b
std::vector<int> Qwen_7b::tokenizer(const std::string& query) {
    auto ids = tokenizer_encode(query);
    // auto prompt = "\n<|im_start|>user\n" + query + "<|im_end|>\n<|im_start|>assistant\n";
    ids.insert(ids.begin(), {198, 151644, 872, 198});
    ids.insert(ids.end(), {151645, 198, 151644, 77091, 198});
    return ids;
}

VARP Qwen_7b::gen_attention_mask(int seq_len) {
    auto attention_mask = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = attention_mask->writeMap<int>();
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            ptr[seq_len * i + j] = j <= i;
        }
    }
    return attention_mask;
}

VARP Qwen_7b::gen_position_ids(int seq_len) {
    auto position_ids = _Input({seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = all_seq_len_;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i;
        }
    }
    return position_ids;
}

bool Qwen_7b::is_stop(int token_id) {
    return token_id >= 151645;
}

// Llama2_7b
std::vector<int> Llama2_7b::tokenizer(const std::string& query) {
    auto ids = tokenizer_encode(query);
    if (model_name_ == "Baichuan2_7b") {
        // baichuan2: <reserved_106>{query}<reserved_107>: 195, query, 196
        ids.insert(ids.begin(), 195);
        ids.push_back(196);
        return ids;
    }
    // llama2: <bos>[INST]{query}[/INST]: 1, 5539, 25580, 29962, query, 12452, 25580, 29962
    ids.insert(ids.begin(), {1, 5539, 25580, 29962});
    ids.insert(ids.end(), {12452, 25580, 29962});
    return ids;
}

VARP Llama2_7b::gen_attention_mask(int seq_len) {
    if (seq_len == 1) {
        auto attention_mask = _Input({1, 1, 1, all_seq_len_ + 1}, NCHW, halide_type_of<float>());
        auto ptr = attention_mask->writeMap<float>();
        for (int i = 0; i < all_seq_len_ + 1; i++) {
            ptr[i] = 0;
        }
        return attention_mask;
    } else {
        auto attention_mask = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<float>());
        auto ptr = attention_mask->writeMap<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                ptr[seq_len * i + j] = (j > i) * std::numeric_limits<float>::lowest();
            }
        }
        return attention_mask;
    }
}

VARP Llama2_7b::gen_position_ids(int seq_len) {
    auto position_ids = _Input({1, seq_len}, NCHW, halide_type_of<int>());
    auto ptr = position_ids->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = all_seq_len_;
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = i;
        }
    }
    return position_ids;
}

bool Llama2_7b::is_stop(int token_id) {
    return token_id == 2;
}
