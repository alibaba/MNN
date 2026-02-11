//
//  diffusion_config.hpp
//  Parse model file paths from config.json (uses rapidjson)
//
//  Separated from diffusion.hpp to keep the base header lightweight.
//
#ifndef MNN_DIFFUSION_CONFIG_HPP
#define MNN_DIFFUSION_CONFIG_HPP

#include <string>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

namespace MNN {
namespace DIFFUSION {

class DiffusionConfig {
public:
    std::string base_dir_;
    rapidjson::Document config_;
    bool config_loaded_ = false;
    
    DiffusionConfig(const std::string& model_path) {
        base_dir_ = model_path;
        if (!base_dir_.empty() && base_dir_.back() != '/') {
            base_dir_ += "/";
        }
        std::string config_path = model_path + "/config.json";
        std::ifstream config_file(config_path);
        if (config_file.is_open()) {
            rapidjson::IStreamWrapper isw(config_file);
            config_.ParseStream(isw);
            if (!config_.HasParseError() && config_.IsObject()) {
                config_loaded_ = true;
            }
        }
    }
    
    std::string text_encoder_model() const {
        if (!config_loaded_) return base_dir_ + "text_encoder.mnn";
        if (config_.HasMember("text_encoder") && config_["text_encoder"].IsObject()) {
            const auto& te = config_["text_encoder"];
            std::string prefix = "";
            if (te.HasMember("directory") && te["directory"].IsString()) {
                std::string dir = te["directory"].GetString();
                if (!dir.empty()) prefix = dir + "/";
            }
            if (te.HasMember("llm") && te["llm"].IsObject()) {
                const auto& llm = te["llm"];
                if (llm.HasMember("model") && llm["model"].IsString())
                    return base_dir_ + prefix + llm["model"].GetString();
            }
            if (te.HasMember("model") && te["model"].IsString())
                return base_dir_ + prefix + te["model"].GetString();
        }
        return base_dir_ + "text_encoder.mnn";
    }
    
    std::string unet_model() const {
        if (!config_loaded_) return base_dir_ + "unet.mnn";
        if (config_.HasMember("transformer") && config_["transformer"].IsObject()) {
            const auto& t = config_["transformer"];
            if (t.HasMember("model") && t["model"].IsString())
                return base_dir_ + t["model"].GetString();
        }
        if (config_.HasMember("unet") && config_["unet"].IsObject()) {
            const auto& u = config_["unet"];
            if (u.HasMember("model") && u["model"].IsString())
                return base_dir_ + u["model"].GetString();
        }
        return base_dir_ + "unet.mnn";
    }
    
    std::string vae_decoder_model() const {
        if (!config_loaded_) return base_dir_ + "vae_decoder.mnn";
        if (config_.HasMember("vae") && config_["vae"].IsObject()) {
            const auto& v = config_["vae"];
            if (v.HasMember("decoder_model") && v["decoder_model"].IsString())
                return base_dir_ + v["decoder_model"].GetString();
        }
        return base_dir_ + "vae_decoder.mnn";
    }
    
    std::string vae_encoder_model() const {
        if (!config_loaded_) return base_dir_ + "vae_encoder.mnn";
        if (config_.HasMember("vae") && config_["vae"].IsObject()) {
            const auto& v = config_["vae"];
            if (v.HasMember("encoder_model") && v["encoder_model"].IsString())
                return base_dir_ + v["encoder_model"].GetString();
        }
        return base_dir_ + "vae_encoder.mnn";
    }
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_DIFFUSION_CONFIG_HPP
