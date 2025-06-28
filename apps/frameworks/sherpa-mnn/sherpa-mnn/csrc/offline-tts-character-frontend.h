// sherpa-mnn/csrc/offline-tts-character-frontend.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_CHARACTER_FRONTEND_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_CHARACTER_FRONTEND_H_
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "sherpa-mnn/csrc/offline-tts-frontend.h"
#include "sherpa-mnn/csrc/offline-tts-vits-model-meta-data.h"

namespace sherpa_mnn {

class OfflineTtsCharacterFrontend : public OfflineTtsFrontend {
 public:
  OfflineTtsCharacterFrontend(const std::string &tokens,
                              const OfflineTtsVitsModelMetaData &meta_data);

  template <typename Manager>
  OfflineTtsCharacterFrontend(Manager *mgr, const std::string &tokens,
                              const OfflineTtsVitsModelMetaData &meta_data);

  /** Convert a string to token IDs.
   *
   * @param text The input text.
   *             Example 1: "This is the first sample sentence; this is the
   *             second one." Example 2: "这是第一句。这是第二句。"
   * @param voice Optional. It is for espeak-ng.
   *
   * @return Return a vector-of-vector of token IDs. Each subvector contains
   *         a sentence that can be processed independently.
   *         If a frontend does not support splitting the text into
   * sentences, the resulting vector contains only one subvector.
   */
  std::vector<TokenIDs> ConvertTextToTokenIds(
      const std::string &text, const std::string &voice = "") const override;

 private:
  OfflineTtsVitsModelMetaData meta_data_;
  std::unordered_map<char32_t, int32_t> token2id_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_CHARACTER_FRONTEND_H_
