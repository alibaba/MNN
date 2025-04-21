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
#include <audio/audio.hpp>

namespace mls {

std::string trimLeadingWhitespace(const std::string& str) {
    auto it = std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch); // Find the first non-whitespace character
    });
    return {it, str.end()}; // Create a substring from the first non-whitespace character
}

const char* getUserString(const char* user_content, bool for_history, bool is_r1) {
    if (is_r1) {
        return ("<|User|>" + std::string(user_content) + "<|Assistant|>" + (for_history ? "" : "<think>\n")).c_str();
    } else {
        return user_content;
    }
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

LlmSession::LlmSession(std::string model_path, json config, std::vector<std::string> history):
    model_path_(std::move(model_path)), config_(std::move(config)) {
    keep_history_ = !config_.contains("keep_history") || config_["keep_history"].get<bool>();
    is_r1_ = config_.contains("is_r1") && config_["is_r1"].get<bool>();
    history_.emplace_back("system", is_r1_ ?
    "<|begin_of_sentence|>You are a helpful assistant." :
    "You are a helpful assistant.");
    if (!history.empty()) {
        for (int i = 0; i < history.size(); i++) {
            if (is_r1_) {
                if (i % 2 == 0) {
                    history_.emplace_back("user", getUserString(history[i].c_str(), true, is_r1_));
                } else {
                    history_.emplace_back("assistant", getR1AssistantString(history[i]));
                }
            } else {
                history_.emplace_back(i % 2 == 0 ? "user" : "assistant", history[i]);
            }
        }
    }
}

void LlmSession::Load() {
    std::string root_cache_dir_str = config_["mmap_dir"];
    bool use_mmap = !config_["mmap_dir"].get<std::string>().empty();
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);
    llm_ = Llm::createLLM(model_path_);
    json extra_config;
    extra_config["use_mmap"] = use_mmap;
    if (use_mmap) {
        std::string temp_dir = root_cache_dir_str;
        extra_config["tmp_path"] = temp_dir;
    }
    if (is_r1_) {
        extra_config["use_template"] = false;
        extra_config["precision"] = "high";
    }
    auto extra_config_str = extra_config.dump();
    MNN_DEBUG("extra_config: %s", extra_config_str.c_str());
//    llm_->set_config(extra_config_str);
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
            history_.emplace_back("assistant", response_result);
        }
        if (on_progress) {
            bool user_stop_requested = on_progress(utf8Char, is_eop);
            stop_requested_ = is_eop || user_stop_requested;
        }
    });
    LlmStreamBuffer stream_buffer{[&processor](const char* str, size_t len){
        processor.processStream(str, len);
    }};
    std::ostream output_ostream(&stream_buffer);
    history_.emplace_back("user", getUserString(prompt.c_str(), false, is_r1_));
    MNN_DEBUG("submitNative history count %zu", history_.size());
    for (auto & it : history_) {
        prompt_string_for_debug += it.second;
    }
    MNN_DEBUG("submitNative prompt_string_for_debug count %s", prompt_string_for_debug.c_str());
    llm_->response(prompt, &output_ostream, "<eop>", 9999);
//    llm_->response(history_, &output_ostream, "<eop>", 1);
//    while (!stop_requested_) {
//        llm_->generate(1);
//    }
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
            if (callback) {
                return callback(ptr, size, last_chunk);
            }
            return true;
        });
    } else {
        MNN_ERROR("no llm instance");
    }
}

}