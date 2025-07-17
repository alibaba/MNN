#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "wavfile.hpp"
#include "mnn_tts_config.hpp"
//#include "piper/mnn_piper_tts_impl.hpp"
#include "bertvits2/mnn_bertvits2_tts_impl.hpp"

class MNNTTSSDK
{
public:
  MNNTTSSDK(const std::string &config_folder);

  // synthesize audio
  std::tuple<int, Audio> Process(const std::string &text);
  void WriteAudioToFile(const Audio &audio_data, const std::string &output_file_path);

private:
  int sample_rate_;
  std::shared_ptr<MNNTTSImplBase> impl_;
};