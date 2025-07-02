
#include "mnn_tts_sdk.hpp"
#include "piper/utf8.h"
#include <mutex>
#include <codecvt> // For std::wstring_convert and std::codecvt_utf8
#include <locale>

MNNTTSSDK::MNNTTSSDK(const std::string &config_folder)
{
  std::string config_json_path = config_folder + "/config.json";
  auto config = MNNTTSConfig(config_json_path);
  auto model_type = config.model_type_;
  auto model_path = config_folder + "/" + config.model_path_;
  auto assset_folder = config_folder + "/" + config.asset_folder_;
  auto cache_folder = config_folder + "/" + config.cache_folder_;
  sample_rate_ = config.sample_rate_;

  if (model_type == "piper")
  {
    impl_ = nullptr;
//            std::make_shared<MNNPiperTTSImpl>(assset_folder, model_path, cache_folder);
  }
  else if (model_type == "bertvits")
  {
    impl_ = std::make_shared<MNNBertVits2TTSImpl>(assset_folder, model_path, cache_folder);
  }
  else
  {
    throw std::runtime_error("Invalid model type");
    return;
  }
}
std::tuple<int, Audio> MNNTTSSDK::Process(const std::string &text)
{
  return impl_->Process(text);
}

void MNNTTSSDK::WriteAudioToFile(const Audio &audio_data, const std::string &output_file_path)
{
  std::ofstream audioFile(output_file_path, std::ios::binary);

  // Write WAV
  writeWavHeader(sample_rate_, 2, 1, (int32_t)audio_data.size(), audioFile);

  audioFile.write((const char *)audio_data.data(),
                  sizeof(int16_t) * audio_data.size());
}
