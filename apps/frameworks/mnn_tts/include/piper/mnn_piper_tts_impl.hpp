#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <mutex> // Include mutex header

#include "mnn_tts_impl_base.hpp"
#include "audio_generator.hpp"
#include "phoneme_ids.hpp"
#include "espeak_ng_wrapper.hpp" // Use wrapper for espeak-ng
#include "mnn_tts_logger.hpp"
#include "uni_algo.hpp"

typedef std::chrono::milliseconds ms;
using clk = std::chrono::system_clock;

typedef std::vector<int16_t> Audio;

class MNNPiperTTSImpl: public MNNTTSImplBase
{
public:
  // Must be called before using textTo* functions
  MNNPiperTTSImpl(const std::string &espeak_data_path, const std::string &model_path, const std::string &cache_path);

  void phonemize_eSpeak(std::string text, std::vector<std::vector<Phoneme>> &phonemes);

  std::vector<int16_t> synthesize(std::vector<PhonemeId> &phonemeIds);

  // Phonemize text and synthesize audio
  std::tuple<int, Audio> Process(const std::string& text) override;

private:
  std::mutex mtx_; // Mutex for thread safety
  AudioGenerator audio_generator_;
  PhonemeIdMap phone_id_map_;
  int sample_rate_ = 16000;
  Phoneme period = U'.';      // CLAUSE_PERIOD
  Phoneme comma = U',';       // CLAUSE_COMMA
  Phoneme question = U'?';    // CLAUSE_QUESTION
  Phoneme exclamation = U'!'; // CLAUSE_EXCLAMATION
  Phoneme colon = U':';       // CLAUSE_COLON
  Phoneme semicolon = U';';   // CLAUSE_SEMICOLON
  Phoneme space = U' ';
};