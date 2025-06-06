//
// Created by ruoyi.sjd on 2025/4/18.
//

#include "llm_session.h"
#include <utility>
#include "MNN/MNNForwardType.h"
#include "MNN/expr/ExecutorScope.hpp"
#include "mls_log.h"
#include "mls_config.h"
#include "utf8_stream_processor.hpp"
#include "llm_stream_buffer.hpp"

namespace mls {

std::string trimLeadingWhitespace(const std::string& str) {
    auto it = std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    });
    return {it, str.end()};
}

const char* getUserString(const char* user_content, bool for_history, bool is_r1) {
    if (is_r1) {
        return ("<|User|>" + std::string(user_content) + "<|Assistant|>" + (for_history ? "" : "<think>\n")).c_str();
    } else {
        return user_content;
    }
}

std::string GetSystemPromptString(std::string system_prompt,  bool is_r1) {
    if (is_r1) {
        return std::string("<|begin_of_sentence|>") + system_prompt;
    } else {
        return system_prompt;
    }
}

std::string deleteThinkPart(std::string assistant_content) {
    std::size_t think_start = assistant_content.find("<think>");
    if (think_start == std::string::npos) {
        return assistant_content;
    }
    std::size_t think_end = assistant_content.find("</think>", think_start);
    if (think_end == std::string::npos) {
        return assistant_content;
    }
    think_end += std::string("</think>").length();
    assistant_content.erase(think_start, think_end - think_start);
    return assistant_content;
}

std::string getR1AssistantString(std::string assistant_content) {
    std::size_t pos = assistant_content.find("</think>");
    if (pos != std::string::npos) {
        assistant_content.erase(0, pos + std::string("</think>").length());
    }
    return trimLeadingWhitespace(assistant_content) + "<|end_of_sentence|>";
}

void LlmSession::Reset() {
    history_.resize(1);
}

LlmSession::LlmSession(std::string model_path, json config, json extra_config, std::vector<std::string> history):
        model_path_(std::move(model_path)), config_(std::move(config)), extra_config_(std::move(extra_config)) {
    max_new_tokens_ = config_.contains("max_new_tokens") ?  config_["max_new_tokens"].get<int>() : 2048;
    keep_history_ = !extra_config_.contains("keep_history") || extra_config_["keep_history"].get<bool>();
    is_r1_ = extra_config_.contains("is_r1") && extra_config_["is_r1"].get<bool>();
    system_prompt_ = config_.contains("system_prompt") ? config_["system_prompt"].get<std::string>() : "You are a helpful assistant.";
    history_.emplace_back("system", GetSystemPromptString(system_prompt_, is_r1_));
    if (!history.empty()) {
        for (int i = 0; i < history.size(); i++) {
            if (is_r1_) {
                if (i % 2 == 0) {
                    history_.emplace_back("user", getUserString(history[i].c_str(), true, is_r1_));
                } else {
                    history_.emplace_back("assistant", getR1AssistantString(history[i]));
                }
            } else {
                history_.emplace_back(i % 2 == 0 ? "user" : "assistant",
                                      i % 2 == 0 ? history[i] :
                                      deleteThinkPart(history[i]));
            }
        }
    }
}

void LlmSession::Load() {
    std::string root_cache_dir_str = extra_config_["mmap_dir"];
    bool use_mmap = !extra_config_["mmap_dir"].get<std::string>().empty();
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);
    llm_ = Llm::createLLM(model_path_);
    json config = config_;
    config["use_mmap"] = use_mmap;
    if (use_mmap) {
        std::string temp_dir = root_cache_dir_str;
        config["tmp_path"] = temp_dir;
    }
    if (is_r1_) {
        config["use_template"] = false;
        config["precision"] = "high";
    }
    current_config_ = config;
    auto config_str = config.dump();
    MNN_DEBUG("extra_config: %s", config_str.c_str());
    llm_->set_config(config_str);
    MNN_DEBUG("dumped config: %s", llm_->dump_config().c_str());
    llm_->load();
}

LlmSession::~LlmSession() {
    delete llm_;
}

const MNN::Transformer::LlmContext * LlmSession::Response(const std::string &prompt,
                                                          const std::function<bool(const std::string&, bool is_eop)>& on_progress) {
    if (llm_ == nullptr) {
        return nullptr;
    }

    if (!keep_history_) {
        history_.resize(1);
    }
    int current_size = 0;
    stop_requested_ = false;
    generate_text_end_ = false;
    std::stringstream response_buffer;
    mls::Utf8StreamProcessor processor([&response_buffer, &on_progress, this](const std::string& utf8Char) {
        bool is_eop = utf8Char.find("<eop>") != std::string::npos;
        if (!is_eop) {
            response_buffer << utf8Char;
        } else {
            std::string response_result =  response_buffer.str();
            MNN_DEBUG("submitNative Result %s", response_result.c_str());
            response_string_for_debug = response_result;
            if (is_r1_) {
                auto& last_message = history_.at(history_.size() - 1);
                std::size_t user_think_pos = last_message.second.find("<think>\n");
                if (user_think_pos != std::string::npos) {
                    last_message.second.erase(user_think_pos, std::string("<think>\n").length());
                }
                response_result = getR1AssistantString(response_result);
            }
            response_result = trimLeadingWhitespace(deleteThinkPart(response_result));
            history_.emplace_back("assistant", response_result);
        }
        if (on_progress) {
            bool user_stop_requested = on_progress(utf8Char, is_eop);
            generate_text_end_ = is_eop;
            stop_requested_ = user_stop_requested;
        }
    });
    LlmStreamBuffer stream_buffer{[&processor](const char* str, size_t len){
        processor.processStream(str, len);
    }};
    std::ostream output_ostream(&stream_buffer);
//#define USE_DEBUG_PROMOT
#ifdef USE_DEBUG_PROMOT
    std::string debug_prompt = "<audio>/data/user/0/com.alibaba.mnnllm.android/files/history/1746690738111/record_1746690751335.wav</audio>";
        history_.emplace_back("user", getUserString(debug_prompt.c_str(), false, is_r1_));
#else
    history_.emplace_back("user", getUserString(prompt.c_str(), false, is_r1_));
#endif
    MNN_DEBUG("submitNative history count %zu", history_.size());
    for (auto & it : history_) {
        prompt_string_for_debug += it.second;
    }
    MNN_DEBUG("submitNative prompt_string_for_debug count %s max_new_tokens_:%d", prompt_string_for_debug.c_str(), max_new_tokens_);
    llm_->response(history_, &output_ostream, "<eop>", 1);
    current_size++;
    while (!stop_requested_ && !generate_text_end_ && current_size < max_new_tokens_) {
        llm_->generate(1);
        current_size++;
    }
    if (!stop_requested_ && enable_audio_output_) {
        llm_->generateWavform();
    }
    auto context = llm_->getContext();
    return context;
}

std::string LlmSession::getDebugInfo() {
    return ("last_prompt:\n" + prompt_string_for_debug + "\nlast_response:\n" + response_string_for_debug);
}

void LlmSession::SetWavformCallback(std::function<bool(const float *, size_t, bool)> callback) {
    if (llm_ != nullptr && callback != nullptr) {
        waveform.clear();
        llm_->setWavformCallback([this, callback = std::move(callback)](const float *ptr, size_t size, bool last_chunk) {
#if DEBUG_SAVE_WAV
            waveform.reserve(waveform.size() + size);
            waveform.insert(waveform.end(), ptr, ptr + size);
            MNN_DEBUG("waveform size %zu", waveform.size());
            if (last_chunk) {
                auto waveform_var = MNN::Express::_Const(waveform.data(), {(int)waveform.size()}, MNN::Express::NCHW, halide_type_of<float>());
                MNN::AUDIO::save("/data/data/com.alibaba.mnnllm.android/files/output.wav", waveform_var, 24000);
                waveform.clear();
            }
#endif
            if (!enable_audio_output_ || stop_requested_) {
                return false;
            }
            if (callback) {
                return !callback(ptr, size, last_chunk);
            }
            return false;
        });
    } else {
        MNN_ERROR("no llm instance");
    }
}

void LlmSession::SetMaxNewTokens(int i) {
    max_new_tokens_ = i;
}

void LlmSession::setSystemPrompt(std::string system_prompt) {
    system_prompt_= std::move(system_prompt);
    if (history_.size() > 1) {
        history_.at(0).second = GetSystemPromptString(system_prompt_, is_r1_);
    } else {
        history_.emplace_back("system", GetSystemPromptString(system_prompt_, is_r1_));
    }
}

void LlmSession::SetAssistantPrompt(const std::string& assistant_prompt) {
    current_config_["assistant_prompt_template"] = assistant_prompt;
    if (llm_) {
        llm_->set_config(current_config_.dump());
    }
    MNN_DEBUG("dumped config: %s", llm_->dump_config().c_str());
}

void LlmSession::enableAudioOutput(bool enable) {
    enable_audio_output_ = enable;
}


    const MNN::Transformer::LlmContext * LlmSession::ResponseWithHistory(
            const std::vector<PromptItem>& full_history,
            const std::function<bool(const std::string&, bool is_eop)>& on_progress) {
        if (llm_ == nullptr) {
            return nullptr;
        }


        // 创建临时历史，不修改成员变量
        std::vector<PromptItem> temp_history;

        // 直接使用传入的完整历史，不保存到成员变量
        temp_history.insert(temp_history.end(), full_history.begin(), full_history.end());

        int current_size = 0;
        stop_requested_ = false;
        generate_text_end_ = false;
        std::stringstream response_buffer;

        // 流处理逻辑，但不修改 history_ 成员
        mls::Utf8StreamProcessor processor([&response_buffer, &on_progress, this](const std::string& utf8Char) {
            bool is_eop = utf8Char.find("<eop>") != std::string::npos;
            if (!is_eop) {
                response_buffer << utf8Char;
            } else {
                std::string response_result = response_buffer.str();
                MNN_DEBUG("ResponseWithHistory Result %s", response_result.c_str());
                response_string_for_debug = response_result;
                if (is_r1_) {
                    response_result = getR1AssistantString(response_result);
                }
                response_result = trimLeadingWhitespace(deleteThinkPart(response_result));
                // 注意：这里不再调用 history_.emplace_back() 保存历史
            }
            if (on_progress) {
                bool user_stop_requested = on_progress(utf8Char, is_eop);
                generate_text_end_ = is_eop;
                stop_requested_ = user_stop_requested;
            }
        });

        LlmStreamBuffer stream_buffer{[&processor](const char* str, size_t len){
            processor.processStream(str, len);
        }};
        std::ostream output_ostream(&stream_buffer);



        MNN_DEBUG("submitNative history count %zu", temp_history.size());
        prompt_string_for_debug.clear(); // 清空旧数据以避免重复追加
        for (const auto& it : temp_history) {
            // 添加角色标签和内容，例如 "[user]: Hello"
            prompt_string_for_debug += "[" + it.first + "]: " + it.second + "\n";
        }
        MNN_DEBUG("submitNative prompt_string_for_debug:\n%s\nmax_new_tokens_:%d", prompt_string_for_debug.c_str(), max_new_tokens_);
        // 使用临时历史进行推理
        llm_->response(temp_history, &output_ostream, "<eop>", 1);
        current_size++;

        while (!stop_requested_ && !generate_text_end_ && current_size < max_new_tokens_) {
            llm_->generate(1);
            current_size++;
        }

        if (!stop_requested_ && enable_audio_output_) {
            llm_->generateWavform();
        }

        return llm_->getContext();
    }

    void LlmSession::clearHistory(int numToKeep) {
        if (numToKeep < 0) {
            numToKeep = 0;
        }
        if (history_.size() > static_cast<size_t>(numToKeep)) {
            history_.erase(history_.begin() + numToKeep, history_.end());
        }
        // 清空相关缓存
        prompt_string_for_debug.clear();
        //response_string_for_debug.clear();
    }

    std::string LlmSession::getSystemPrompt() const {
        return system_prompt_;
    }

}