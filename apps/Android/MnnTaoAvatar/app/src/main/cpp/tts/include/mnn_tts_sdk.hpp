#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "wavfile.hpp"
#include "mnn_tts_config.hpp"
#include "mnn_tts_impl_base.hpp"
//#include "piper/mnn_piper_tts_impl.hpp"
#include "bertvits2/mnn_bertvits2_tts_impl.hpp"

class MNNTTSSDK
{
public:
  MNNTTSSDK(const std::string &config_folder, const std::string &params_json = "{}");

  // synthesize audio
  std::tuple<int, Audio> Process(const std::string &text);
  void WriteAudioToFile(const Audio &audio_data, const std::string &output_file_path);
  
  // Set speaker ID dynamically (only for supertonic model)
  void SetSpeakerId(const std::string &speaker_id);

private:
  int sample_rate_;
  std::shared_ptr<MNNTTSImplBase> impl_;
  std::string model_type_;  // Store model type for SetSpeakerId
  
  // 辅助方法：JSON 字符串转 map
  std::map<std::string, std::string> parseJsonToMap(const std::string &json_str);
};
