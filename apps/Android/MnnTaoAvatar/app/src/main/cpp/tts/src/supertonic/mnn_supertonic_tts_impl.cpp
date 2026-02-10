/**
 * @file mnn_supertonic_tts_impl.cpp
 * @brief MNN Supertonic TTS实现类
 */

#include "supertonic/mnn_supertonic_tts_impl.hpp"
#include "mnn_tts_logger.hpp"
#include "utils.hpp"
#include <nlohmann/json.hpp>

#include "piper/uni_algo.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib> // for getenv
#include <fstream>
#include <random>
#include <regex>
#include <unordered_map>
#include <unordered_set>

using json = nlohmann::json;
using namespace MNN::Express;

namespace
{ // Helper namespace

  // Emoji ranges check (matching Python regex)
  bool is_emoji(char32_t cp)
  {
    if (cp >= 0x1F600 && cp <= 0x1F64F)
      return true; // emoticons 
    if (cp >= 0x1F300 && cp <= 0x1F5FF)
      return true; // symbols & pictographs
    if (cp >= 0x1F680 && cp <= 0x1F6FF)
      return true; // transport & map
    if (cp >= 0x1F700 && cp <= 0x1F77F)
      return true;
    if (cp >= 0x1F780 && cp <= 0x1F7FF)
      return true;
    if (cp >= 0x1F800 && cp <= 0x1F8FF)
      return true;
    if (cp >= 0x1F900 && cp <= 0x1F9FF)
      return true;
    if (cp >= 0x1FA00 && cp <= 0x1FA6F)
      return true;
    if (cp >= 0x1FA70 && cp <= 0x1FAFF)
      return true;
    if (cp >= 0x2600 && cp <= 0x26FF)
      return true;
    if (cp >= 0x2700 && cp <= 0x27BF)
      return true;
    if (cp >= 0x1F1E6 && cp <= 0x1F1FF)
      return true; // flags?
    return false;
  }

  // Combining diacritics check
  bool is_combining_diacritic(char32_t cp)
  {
    // [\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030A\u030B\u030C\u0327\u0328\u0329\u032A\u032B\u032C\u032D\u032E\u032F]
    // Simplified range check for performance (approximate 0300-036F block often
    // used) But Python text.py is specific. Let's list them or use range if
    // contiguous. They are mostly in 0x0300 block.
    static const std::unordered_set<char32_t> diacritics = {
        0x0302, 0x0303, 0x0304, 0x0305, 0x0306, 0x0307, 0x0308,
        0x030A, 0x030B, 0x030C, 0x0327, 0x0328, 0x0329, 0x032A,
        0x032B, 0x032C, 0x032D, 0x032E, 0x032F};
    return diacritics.count(cp);
  }

  // Special symbols check
  bool is_special_symbol(char32_t cp)
  {
    // [♥☆♡©\\]
    return cp == 0x2665 || cp == 0x2606 || cp == 0x2661 || cp == 0x00A9 ||
           cp == 0x005C;
  }

  bool is_end_punctuation(char32_t cp)
  {
    // [.!?;:,'\"')\]}…。」』】〉》›»]
    static const std::unordered_set<char32_t> puncts = {
        '.', '!', '?', ';', ':', ',', '\'',
        '"', ')', ']', '}', 0x2026, 0x3002, 0x300D,
        0x300F, 0x3011, 0x3009, 0x300B, 0x203A, 0x00BB};
    return puncts.count(cp);
  }

  void utf8_append(std::string &s, char32_t cp)
  {
    if (cp < 0x80)
    {
      s.push_back(static_cast<char>(cp));
    }
    else if (cp < 0x800)
    {
      s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else if (cp < 0x10000)
    {
      s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else if (cp <= 0x10FFFF)
    {
      s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

} // namespace

// Text processor implementation
MNNSupertonicTTSImpl::TextProcessor::TextProcessor(
    const std::string &indexer_path)
{
  std::ifstream file(indexer_path);
  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open unicode indexer file: " +
                             indexer_path);
  }

  json indexer_json;
  file >> indexer_json;
  file.close();

  // Parse indexer
  for (auto &item : indexer_json.items())
  {
    uint16_t unicode_val = static_cast<uint16_t>(std::stoi(item.key()));
    int index = item.value().get<int>();
    unicode_to_index_[unicode_val] = index;
    index_to_unicode_[index] = unicode_val;
  }

  PLOG(INFO, "Loaded unicode indexer with " +
                 std::to_string(unicode_to_index_.size()) + " entries");
}

std::vector<int>
MNNSupertonicTTSImpl::TextProcessor::encode(const std::string &text)
{
  std::vector<int> encoded;

  // 1. Normalize NFKD
  std::string text_norm = una::norm::to_nfkd_utf8(text);

  // 2. Filter chars (Emojis, Diacritics, Special Symbols) and Map chars
  std::string filtered;
  filtered.reserve(text_norm.size());

  auto view = una::ranges::utf8_view(text_norm);
  for (auto it = view.begin(); it != view.end(); ++it)
  {
    char32_t cp = *it;

    if (is_emoji(cp))
      continue;
    if (is_combining_diacritic(cp))
      continue;
    if (is_special_symbol(cp))
      continue;

    // Char replacements
    switch (cp)
    {
    case 0x2013: // –
    case 0x2011: // ‑
    case 0x2014: // —
      filtered.push_back('-');
      break;
    case 0x00AF: // ¯
    case 0x005F: // _
    case 0x005B: // [
    case 0x005D: // ]
    case 0x007C: // |
    case 0x002F: // /
    case 0x0023: // #
    case 0x2192: // →
    case 0x2190: // ←
      filtered.push_back(' ');
      break;
    case 0x201C: // “
    case 0x201D: // ”
      filtered.push_back('"');
      break;
    case 0x2018: // ‘
    case 0x2019: // ’
    case 0x00B4: // ´
    case 0x0060: // `
      filtered.push_back('\'');
      break;
    default:
      // Append as UTF-8
      std::string s;
      utf8_append(s, cp);
      filtered += s;
      break;
    }
  }

  std::string t = filtered;

  // 3. String replacements (Expression replacements)
  // "e.g.," -> "for example, "
  // "i.e.," -> "that is, "
  // "@" -> " at "
  // Simple find/replace loop for these few replacements.
  auto replace_all = [](std::string &str, const std::string &from,
                        const std::string &to)
  {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
      str.replace(start_pos, from.length(), to);
      start_pos += to.length();
    }
  };

  replace_all(t, "@", " at ");
  replace_all(t, "e.g.,", "for example, ");
  replace_all(t, "i.e.,", "that is, ");

  // 4. Regex replacements
  // Spacing around punctuation
  t = std::regex_replace(t, std::regex(" ,"), ",");
  t = std::regex_replace(t, std::regex(" \\."), ".");
  t = std::regex_replace(t, std::regex(" !"), "!");
  t = std::regex_replace(t, std::regex(" \\?"), "?");
  t = std::regex_replace(t, std::regex(" ;"), ";");
  t = std::regex_replace(t, std::regex(" :"), ":");
  t = std::regex_replace(t, std::regex(" '"), "'");

  // Duplicate quotes
  replace_all(t, "\"\"", "\"");
  while (t.find("\"\"") != std::string::npos)
    replace_all(t, "\"\"", "\"");
  while (t.find("''") != std::string::npos)
    replace_all(t, "''", "'");
  while (t.find("``") != std::string::npos)
    replace_all(t, "``", "`");

  // Remove extra spaces sequence
  t = std::regex_replace(t, std::regex("\\s+"), " ");

  // Strip leading/trailing whitespace
  t.erase(0, t.find_first_not_of(" "));
  t.erase(t.find_last_not_of(" ") + 1);

  // 5. Add period if needed
  bool has_end_punct = false;
  if (!t.empty())
  {
    auto last_view = una::ranges::utf8_view(t);
    // get last char
    char32_t last_cp = 0;
    for (auto cp : last_view)
      last_cp = cp;
    if (is_end_punctuation(last_cp))
      has_end_punct = true;
  }

  if (!t.empty() && !has_end_punct)
  {
    t += ".";
  }

  // 6. Encode to IDs
  auto final_view = una::ranges::utf8_view(t);
  for (char32_t cp : final_view)
  {
    uint16_t val =
        static_cast<uint16_t>(cp); // Potential truncation if outside BMP
    auto it = unicode_to_index_.find(val);
    if (it != unicode_to_index_.end())
    {
      encoded.push_back(it->second);
    }
    else
    {
      // Handle unknown characters by mapping to index 0 (padding/unknown)
      encoded.push_back(0);
    }
  }

  return encoded;
}

// MNNSupertonicTTSImpl implementation
MNNSupertonicTTSImpl::MNNSupertonicTTSImpl(
    const std::string &models_dir,
    const std::map<std::string, std::string> &overrides)
    : models_dir_(models_dir)
{
  PLOG(INFO, "Initializing Supertonic TTS with models_dir: " + models_dir_);

  // Load config.json to get precision, speaker_id, iter_steps, speed
  std::string config_json_path = models_dir_ + "/config.json";
  json config_json;
  
  // Try to read config.json
  std::ifstream config_json_file(config_json_path);
  if (config_json_file.is_open()) {
    try {
      config_json_file >> config_json;
      config_json_file.close();
    } catch (const std::exception &e) {
      PLOG(WARNING, "Failed to parse config.json: " + std::string(e.what()));
    }
  } else {
    PLOG(WARNING, "config.json not found, using defaults");
  }

  // Get precision: from overrides, then config.json, then default
  if (overrides.find("precision") != overrides.end() && !overrides.at("precision").empty()) {
    precision_dir_ = overrides.at("precision");
  } else if (config_json.contains("precision") && config_json["precision"].is_string()) {
    precision_dir_ = config_json["precision"].get<std::string>();
  } else {
    precision_dir_ = "fp16"; // default
  }

  // Get speaker_id: from overrides, then config.json, then default
  if (overrides.find("speaker_id") != overrides.end() && !overrides.at("speaker_id").empty()) {
    speaker_id_ = overrides.at("speaker_id");
  } else if (config_json.contains("speaker_id") && config_json["speaker_id"].is_string()) {
    speaker_id_ = config_json["speaker_id"].get<std::string>();
  } else {
    speaker_id_ = "M1"; // default
  }

  // Get iter_steps: from overrides, then config.json, then default
  if (overrides.find("iter_steps") != overrides.end() && !overrides.at("iter_steps").empty()) {
    try {
      iter_steps_ = std::stoi(overrides.at("iter_steps"));
    } catch (const std::exception &e) {
      PLOG(WARNING, "Failed to parse iter_steps from overrides, using default");
      iter_steps_ = 10; // default
    }
  } else if (config_json.contains("iter_steps") && config_json["iter_steps"].is_number_integer()) {
    iter_steps_ = config_json["iter_steps"].get<int>();
  } else {
    iter_steps_ = 10; // default
  }

  // Get speed: from overrides, then config.json, then default
  if (overrides.find("speed") != overrides.end() && !overrides.at("speed").empty()) {
    try {
      speed_ = std::stof(overrides.at("speed"));
    } catch (const std::exception &e) {
      PLOG(WARNING, "Failed to parse speed from overrides, using default");
      speed_ = 1.0f; // default
    }
  } else if (config_json.contains("speed") && config_json["speed"].is_number_float()) {
    speed_ = config_json["speed"].get<float>();
  } else {
    speed_ = 1.0f; // default
  }

  std::cout << "model_dir_" << models_dir_ << std::endl;
  std::cout << "precsion_dir: " << precision_dir_ << std::endl;
  std::cout << "speaker_id: " << speaker_id_ << std::endl;
  std::cout << "iter_steps: " << iter_steps_ << std::endl;
  std::cout << "speed: " << speed_ << std::endl;
  std::cout << "cache_dir_: " << cache_dir_ << std::endl;

  // Load tts.json config
  std::string config_path = models_dir_ + "/mnn_models/tts.json";
  std::ifstream config_file(config_path);
  if (!config_file.is_open())
  {
    PLOG(ERROR, "Failed to open config file: " + config_path);
    throw std::runtime_error("Failed to open config file: " + config_path);
  }
  json config;
  config_file >> config;

  // Parse configuration
  try
  {
    if (config.contains("ae"))
    {
      sample_rate_ = config["ae"].value("sample_rate",
                                        24000); // Default to 24000 if missing
      base_chunk_size_ = config["ae"].value("base_chunk_size", 512);
    }
    else
    {
      sample_rate_ = 24000;
      base_chunk_size_ = 512;
    }

    // Attempt to retrieve TTL configuration
    if (config.contains("ttl"))
    {
      chunk_compress_factor_ = config["ttl"].value("chunk_compress_factor", 6);
      ldim_ = config["ttl"].value("latent_dim", 24);
    }
    else
    {
      // Fallback: Use default values or check specific nested keys
      chunk_compress_factor_ = 6;
      ldim_ = 24;

      if (config.contains("style_encoder") &&
          config["style_encoder"].contains("proj_in"))
      {
        chunk_compress_factor_ = config["style_encoder"]["proj_in"].value(
            "chunk_compress_factor", 6);
        ldim_ = config["style_encoder"]["proj_in"].value("ldim", 24);
      }
    }

    PLOG(INFO, "Config loaded: sample_rate=" + std::to_string(sample_rate_) +
                   ", base_chunk_size=" + std::to_string(base_chunk_size_) +
                   ", chunk_compress_factor=" +
                   std::to_string(chunk_compress_factor_) +
                   ", ldim=" + std::to_string(ldim_));
  }
  catch (const std::exception &e)
  {
    PLOG(ERROR, "Error parsing config: " + std::string(e.what()));
    throw;
  }

  // Initialize text processor
  std::string indexer_path = models_dir_ + "/mnn_models/unicode_indexer.json";
  text_processor_ = std::make_unique<TextProcessor>(indexer_path);

  // Initialize MNN inference engine
  initializeModels();

  // Load Voice Styles
  loadVoiceStyles();

  PLOG(INFO, "Supertonic TTS initialized successfully");
}

void MNNSupertonicTTSImpl::loadVoiceStyles()
{
  PLOG(INFO, "Loading voice styles...");

  // Default voices
  for (const auto &id : voice_ids_)
  {
    loadVoiceStyle(id);
  }
  PLOG(INFO, "Voice styles loaded successfully");
}

void MNNSupertonicTTSImpl::loadVoiceStyle(const std::string &voice_name)
{
  try
  {
    std::string style_path =
        models_dir_ + "/mnn_models/voice_styles/" + voice_name + ".json";
    // Check if file exists, if not try old path structure
    std::ifstream f_check(style_path);
    if (!f_check.good())
    {
      style_path = models_dir_ + "/voice_styles/" + voice_name + ".json";
    }
    f_check.close();

    std::ifstream style_file(style_path);
    if (!style_file.is_open())
    {
      PLOG(WARNING, "Failed to open style.json for voice: " + voice_name +
                        " at " + style_path);
      return;
    }

    json style_json;
    style_file >> style_json;
    style_file.close();

    std::vector<std::vector<float>> ttl_vectors, dp_vectors;

    if (style_json.contains("style_ttl") && style_json.contains("style_dp"))
    {
      // Parse TTL
      for (const auto &ttl_item : style_json["style_ttl"]["data"])
      {
        for (const auto &vec : ttl_item)
        {
          std::vector<float> ttl_vector;
          for (const auto &val : vec)
            ttl_vector.push_back(val.get<float>());
          ttl_vectors.push_back(ttl_vector);
        }
      }
      // Parse DP
      for (const auto &dp_item : style_json["style_dp"]["data"])
      {
        for (const auto &vec : dp_item)
        {
          std::vector<float> dp_vector;
          for (const auto &val : vec)
            dp_vector.push_back(val.get<float>());
          dp_vectors.push_back(dp_vector);
        }
      }
      voice_styles_[voice_name] = VoiceStyle(ttl_vectors, dp_vectors);
      PLOG(INFO, "Loaded voice style: " + voice_name);
    }
  }
  catch (const std::exception &e)
  {
    PLOG(ERROR, "Error loading voice style " + voice_name + ": " + e.what());
  }
}

std::string MNNSupertonicTTSImpl::preprocessText(const std::string &text)
{
  // Remove excess whitespace
  std::string processed = std::regex_replace(text, std::regex("\\s+"), " ");
  // Trim leading/trailing whitespace
  processed = std::regex_replace(processed, std::regex("^\\s+|\\s+$"), "");
  processed = std::regex_replace(processed, std::regex("[\\.!?]+$"), ".");
  return processed;
}

std::tuple<int, Audio> MNNSupertonicTTSImpl::Process(const std::string &text)
{
  const std::string &voice_name = speaker_id_;
  int steps = iter_steps_;
  float speed = speed_; 
  
  if (voice_styles_.find(voice_name) == voice_styles_.end())
  {
    PLOG(ERROR, "Voice style not found: " + voice_name);
    throw std::runtime_error("Voice style not found: " + voice_name);
  }
  std::string processed_text = preprocessText(text);
  return synthesize(processed_text, voice_styles_[voice_name], steps, speed);
}

void MNNSupertonicTTSImpl::SetSpeakerId(const std::string &speaker_id)
{
  // Validate speaker_id exists in voice_styles_
  if (voice_styles_.find(speaker_id) == voice_styles_.end())
  {
    PLOG(ERROR, "Cannot set speaker_id to invalid value: " + speaker_id);
    throw std::runtime_error("Invalid speaker_id: " + speaker_id);
  }
  
  speaker_id_ = speaker_id;
  PLOG(INFO, "Speaker ID changed to: " + speaker_id_);
}

std::tuple<int, Audio>
MNNSupertonicTTSImpl::synthesize(const std::string &text,
                                 const VoiceStyle &voice_style, int steps,
                                 float speed)
{

  auto default_ret = std::make_tuple(sample_rate_, std::vector<int16_t>(0));
  auto start_time = std::chrono::high_resolution_clock::now();

  PLOG(INFO, "Synthesizing text: \"" + text + "\"");

  // 1. Text Encoding
  std::vector<int> text_ids = text_processor_->encode(text);
  // Ensure non-empty text_ids; pad with 0 if necessary
  if (text_ids.empty())
  {
    text_ids.push_back(0);
  }

  // Create text mask (all 1s)
  std::vector<float> text_mask(text_ids.size(), 1.0f);

  // 2. Duration Prediction
  auto duration_outputs = predictDuration(text_ids, voice_style.dp, text_mask);
  if (duration_outputs.empty())
  {
    PLOG(ERROR, "Duration prediction failed");
    return default_ret;
  }

  // 3. Process Duration to get total length
  // The first element of duration_outputs is expected to be the total duration
  float total_duration_sec = duration_outputs[0];

  // Apply speed adjustment
  if (speed > 0.0f)
  {
    total_duration_sec /= speed;
  }

  int wav_len = static_cast<int>(total_duration_sec * sample_rate_);
  int chunk_size = base_chunk_size_ * chunk_compress_factor_;
  int latent_len = (wav_len + chunk_size - 1) / chunk_size;

  // Ensure minimum length
  if (latent_len < 1)
    latent_len = 1;

  wav_len = latent_len * base_chunk_size_; // Recalculate aligned wav_len

  // 4. Generate Text Embedding
  // This uses the text encoder model
  std::vector<float> text_emb =
      encodeText(text_ids, voice_style.ttl, text_mask);
  if (text_emb.empty())
  {
    PLOG(ERROR, "Text encoding failed");
    return default_ret;
  }

  // 5. Vector Estimation (Flow Matching)
  // Flatten style.ttl for Vector Estimator input
  std::vector<float> style_ttl_flat;
  if (!voice_style.ttl.empty())
  {
    int ttl_dim0 = voice_style.ttl.size();
    int ttl_dim1 = voice_style.ttl[0].size();
    style_ttl_flat.reserve(ttl_dim0 * ttl_dim1);
    for (const auto &vec : voice_style.ttl)
    {
      style_ttl_flat.insert(style_ttl_flat.end(), vec.begin(), vec.end());
    }
  }

  // Initialize Latent Mask
  std::vector<int> latent_mask(latent_len, 1);

  // Generate Noisy Latent (Gaussian Noise)
  int latent_dim = ldim_ * chunk_compress_factor_;
  std::vector<float> noisy_latent(1 * latent_dim * latent_len);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dis(0.0f, 1.0f);

  for (size_t i = 0; i < noisy_latent.size(); ++i)
  {
    noisy_latent[i] = dis(gen);
  }

  // Perform Flow Matching steps
  std::vector<float> estimated = noisy_latent;
  for (int step = 0; step < steps; ++step)
  {
    estimated = estimateVector(estimated, text_emb, style_ttl_flat, latent_mask,
                               text_mask, step, steps);
    if (estimated.empty())
    {
      PLOG(ERROR, "Vector estimation failed at step " + std::to_string(step));
      return default_ret;
    }
  }

  // 6. Vocoder Synthesis
  std::vector<float> audio = vocode(estimated);
  if (audio.empty())
  {
    PLOG(ERROR, "Vocoding failed");
    return default_ret;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> duration = end_time - start_time;
  float time_cost = duration.count();
  float audio_duration = static_cast<float>(audio.size()) / sample_rate_;
  float rtf = time_cost / audio_duration;
  PLOG(INFO, "RTF: " + std::to_string(rtf));

  std::vector<int16_t> audio_int16;
  for (int i = 0; i < audio.size(); i++)
  {
    audio_int16.push_back(audio[i] * 32768);
  }

  return std::make_tuple(sample_rate_, audio_int16);
}

// Core processing function
std::vector<float> MNNSupertonicTTSImpl::predictDuration(
    const std::vector<int> &text_ids,
    const std::vector<std::vector<float>> &style_dp,
    const std::vector<float> &text_mask)
{

  MNN::Express::ExecutorScope scope(executor_);

  // Prepare inputs
  std::vector<MNN::Express::VARP> inputs(3);
  int num_tokens = static_cast<int>(text_ids.size());
  int dp_size_0 = static_cast<int>(style_dp.size());
  int dp_size_1 = static_cast<int>(style_dp[0].size());

  // Input 0: text_ids {1, num_tokens} NCHW
  inputs[0] =
      MNN::Express::_Input({1, num_tokens}, NCHW, halide_type_of<int>());
  auto ptr0 = inputs[0]->writeMap<int>();
  for (int i = 0; i < num_tokens; ++i)
  {
    ptr0[i] = text_ids[i];
  }

  // Input 1: style_dp {1, dp_size_0, dp_size_1} NCHW
  inputs[1] = MNN::Express::_Input({1, dp_size_0, dp_size_1}, NCHW,
                                   halide_type_of<float>());
  auto ptr1 = inputs[1]->writeMap<float>();
  for (int i = 0; i < dp_size_0; ++i)
  {
    for (int j = 0; j < dp_size_1; ++j)
    {
      ptr1[i * dp_size_1 + j] = style_dp[i][j];
    }
  }

  // Input 2: text_mask {1, 1, num_tokens} NCHW
  inputs[2] =
      MNN::Express::_Input({1, 1, num_tokens}, NCHW, halide_type_of<float>());
  auto ptr2 = inputs[2]->writeMap<float>();
  for (int i = 0; i < num_tokens; ++i)
  {
    ptr2[i] = text_mask[i];
  }

  // Run Inference
  std::vector<MNN::Express::VARP> outputs = dp_module_->onForward(inputs);

  // Process Output
  if (outputs.empty())
  {
    PLOG(ERROR, "Duration Predictor returned empty output");
    return {};
  }

  auto output = outputs[0];
  auto size = output->getInfo()->size;
  std::vector<float> result(size);
  ::memcpy(result.data(), output->readMap<float>(), size * sizeof(float));

  return result;
}

std::vector<float> MNNSupertonicTTSImpl::encodeText(
    const std::vector<int> &text_ids,
    const std::vector<std::vector<float>> &style_ttl,
    const std::vector<float> &text_mask)
{

  MNN::Express::ExecutorScope scope(executor_);

  std::vector<MNN::Express::VARP> inputs(3);

  // Input 0: Text IDs
  int num_tokens = static_cast<int>(text_ids.size());
  inputs[0] =
      MNN::Express::_Input({1, num_tokens}, NCHW, halide_type_of<int>());
  auto ptr0 = inputs[0]->writeMap<int>();
  for (int i = 0; i < num_tokens; ++i)
  {
    ptr0[i] = text_ids[i];
  }

  // Input 1: Style Output (TTL)
  int ttl_size_0 = static_cast<int>(style_ttl.size());
  int ttl_size_1 = static_cast<int>(style_ttl[0].size());
  inputs[1] = MNN::Express::_Input({1, ttl_size_0, ttl_size_1}, NCHW,
                                   halide_type_of<float>());
  auto ptr1 = inputs[1]->writeMap<float>();
  for (int i = 0; i < ttl_size_0; ++i)
  {
    for (int j = 0; j < ttl_size_1; ++j)
    {
      ptr1[i * ttl_size_1 + j] = style_ttl[i][j];
    }
  }

  // Input 2: Text Mask
  inputs[2] =
      MNN::Express::_Input({1, 1, num_tokens}, NCHW, halide_type_of<float>());
  auto ptr2 = inputs[2]->writeMap<float>();
  for (int i = 0; i < num_tokens; ++i)
  {
    ptr2[i] = text_mask[i];
  }

  // Run Inference
  std::vector<MNN::Express::VARP> outputs = te_module_->onForward(inputs);

  if (outputs.empty())
  {
    PLOG(ERROR, "Text Encoder returned empty output");
    return {};
  }

  auto output = outputs[0];
  auto size = output->getInfo()->size;
  std::vector<float> result(size);
  ::memcpy(result.data(), output->readMap<float>(), size * sizeof(float));

  return result;
}

std::vector<float> MNNSupertonicTTSImpl::estimateVector(
    const std::vector<float> &noisy_latent, const std::vector<float> &text_emb,
    const std::vector<float> &style_ttl, const std::vector<int> &latent_mask,
    const std::vector<float> &text_mask, int current_step, int total_step)
{

  MNN::Express::ExecutorScope scope(executor_);

  // Prepare 7 inputs
  std::vector<MNN::Express::VARP> inputs(7);

  // Shapes derived from mnn_estimator.cpp
  int total_size = static_cast<int>(noisy_latent.size());
  int latent_dim = ldim_ * chunk_compress_factor_;
  int latent_len = total_size / latent_dim;

  // Input 0: noisy_latent {1, latent_dim, latent_len} NCHW
  inputs[0] = MNN::Express::_Input({1, latent_dim, latent_len}, NCHW,
                                   halide_type_of<float>());
  ::memcpy(inputs[0]->writeMap<float>(), noisy_latent.data(),
           total_size * sizeof(float));

  // Input 1: text_emb {1, channels, text_len}
  int text_len = static_cast<int>(text_mask.size());
  int text_channels = text_emb.size() / text_len;

  inputs[1] = MNN::Express::_Input({1, text_channels, text_len}, NCHW,
                                   halide_type_of<float>());
  ::memcpy(inputs[1]->writeMap<float>(), text_emb.data(),
           text_emb.size() * sizeof(float));

  // Input 2: style_ttl {1, num_style_vectors, 256} NCHW
  int style_len = static_cast<int>(style_ttl.size());
  int style_dim = 256;
  int num_style_vectors = style_len / style_dim;

  inputs[2] = MNN::Express::_Input({1, num_style_vectors, style_dim}, NCHW,
                                   halide_type_of<float>());
  ::memcpy(inputs[2]->writeMap<float>(), style_ttl.data(),
           style_ttl.size() * sizeof(float));

  // Input 3: latent_mask {1, 1, latent_len}
  inputs[3] =
      MNN::Express::_Input({1, 1, latent_len}, NCHW, halide_type_of<float>());
  auto ptr3 = inputs[3]->writeMap<float>();
  for (int i = 0; i < latent_len; ++i)
    ptr3[i] = static_cast<float>(latent_mask[i]);

  // Input 4: text_mask {1, 1, text_len}
  inputs[4] =
      MNN::Express::_Input({1, 1, text_len}, NCHW, halide_type_of<float>());
  ::memcpy(inputs[4]->writeMap<float>(), text_mask.data(),
           text_mask.size() * sizeof(float));

  // Input 5: current_step {1}
  inputs[5] = MNN::Express::_Input({1}, NCHW, halide_type_of<float>());
  inputs[5]->writeMap<float>()[0] = static_cast<float>(current_step);

  // Input 6: total_step {1}
  inputs[6] = MNN::Express::_Input({1}, NCHW, halide_type_of<float>());
  inputs[6]->writeMap<float>()[0] = static_cast<float>(total_step);

  auto outputs = ve_module_->onForward(inputs);

  if (outputs.empty())
  {
    PLOG(ERROR, "Vector Estimator returned empty output");
    return {};
  }

  auto output = outputs[0];
  auto size = output->getInfo()->size;
  std::vector<float> result(size);
  ::memcpy(result.data(), output->readMap<float>(), size * sizeof(float));
  return result;
}

std::vector<float>
MNNSupertonicTTSImpl::vocode(const std::vector<float> &latent)
{
  MNN::Express::ExecutorScope scope(executor_);

  // Shape: {1, 144, latent_len}
  // Latent dim 144.
  int total_size = static_cast<int>(latent.size());
  int dim = 144;
  int len = total_size / dim;

  std::vector<MNN::Express::VARP> inputs(1);
  inputs[0] =
      MNN::Express::_Input({1, dim, len}, NCHW, halide_type_of<float>());
  ::memcpy(inputs[0]->writeMap<float>(), latent.data(),
           total_size * sizeof(float));

  auto outputs = vc_module_->onForward(inputs);
  if (outputs.empty())
    return {};

  auto output = outputs[0];
  auto size = output->getInfo()->size;
  std::vector<float> result(size);
  ::memcpy(result.data(), output->readMap<float>(), size * sizeof(float));
  return result;
}

// std::tuple<int, Audio> MNNSupertonicTTSImpl::Process(const std::string &text)
// {
//   // Simplified Process implementation for compatibility with base class
//   // interface
//   VoiceStyle default_voice({{{0.1f, 0.2f, 0.3f}}}, {{{0.4f, 0.5f, 0.6f}}});
//   auto [audio_float, sample_rate, rtf] =
//       synthesize(text, default_voice, 10, 1.0f);

//   // Convert float to int16_t
//   Audio audio_int16(audio_float.size());
//   for (size_t i = 0; i < audio_float.size(); ++i) {
//     audio_int16[i] = static_cast<int16_t>(audio_float[i] * 32767.0f);
//   }

//   return std::make_tuple(sample_rate, audio_int16);
// }

// Initialize MNN Models
void MNNSupertonicTTSImpl::initializeModels()
{
  PLOG(INFO, "Initializing models...");

  // Set up Runtime (Executor) with Low Precision config
  MNN::BackendConfig backendConfig;
  backendConfig.precision = MNN::BackendConfig::Precision_Low;
  backendConfig.memory = MNN::BackendConfig::Memory_Low;

  // Create Executor once
  executor_ = std::shared_ptr<MNN::Express::Executor>(
      MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 4));

  MNN::Express::ExecutorScope scope(executor_);

  // Load Models directly using Module::load
  auto loadModule = [&](const std::string &filename, const std::string &name)
  {
    std::string path =
        models_dir_ + "/mnn_models/" + precision_dir_ + "/" + filename;
    std::vector<std::string> inputs,
        outputs; // Empty for auto-detection or not needed for load
    auto module = std::shared_ptr<MNN::Express::Module>(
        MNN::Express::Module::load(inputs, outputs, path.c_str()));
    if (!module)
    {
      PLOG(ERROR, "Failed to load " + name + ": " + path);
      throw std::runtime_error("Failed to load model: " + path);
    }
    PLOG(INFO, "Successfully loaded " + name);
    return module;
  };

  dp_module_ = loadModule("duration_predictor.mnn", "Duration Predictor");
  te_module_ = loadModule("text_encoder.mnn", "Text Encoder");
  ve_module_ = loadModule("vector_estimator.mnn", "Vector Estimator");
  vc_module_ = loadModule("vocoder.mnn", "Vocoder");
}