#include "tts_service.hpp"
#include <memory>
#include <jni.h>

namespace TaoAvatar {

TTSService::~TTSService() {
    tts_ = nullptr;
}

bool TTSService::LoadTtsResources(const char *resPath, const char* modelName, const char* cacheDir) {
    MH_DEBUG("TTSService::LoadTtsResources resPath: %s", resPath);
    if (!tts_) {
        tts_ = std::make_shared<MNNTTSSDK>(
                std::string(resPath));
    }
    if (!tts_) {
        MH_ERROR("Failed to create TTSService.");
        return false;
    }
    return true;
}

void WriteToFileForDebug(const std::vector<int16_t> &audio, const std::string &file_name) {
    std::ofstream outFile(file_name, std::ios::binary);
    if (outFile.is_open()) {
        size_t size = audio.size();
        outFile.write(reinterpret_cast<char*>(&size), sizeof(size));
        if (!audio.empty()) {
            outFile.write(reinterpret_cast<const char*>(audio.data()),
                          audio.size() * sizeof(int16_t));
        }
        outFile.close();
    }
}


void TTSService::SetIndex(int index) {
    current_index_ = index;
}

void TTSService::SetSpeakerId(const std::string &speaker_id) {
    if (tts_) {
        tts_->SetSpeakerId(speaker_id);
    }
}

std::vector<int16_t> TTSService::Process(const std::string &text, int id) {
    if (tts_ != nullptr && (!text.empty())) {
        auto audio = tts_->Process(text);
#if DEBUG_SAVE_TTS_DATA
        WriteToFileForDebug( std::get<1>(audio),
                "/data/data/com.taobao.meta.avatar/tts_" + std::to_string(id) + ".pcm");
#endif
        return std::get<1>(audio);
    } else {
        MH_ERROR("Failed to process text to speech.");
    }
    return {};
}

TTSService::TTSService(std::string language):language_(std::move(language)) {

}

}
