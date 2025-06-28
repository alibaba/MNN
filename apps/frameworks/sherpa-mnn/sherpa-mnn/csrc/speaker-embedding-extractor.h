// sherpa-mnn/csrc/speaker-embedding-extractor.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-mnn/csrc/online-stream.h"
#include "sherpa-mnn/csrc/parse-options.h"

namespace sherpa_mnn {

struct SpeakerEmbeddingExtractorConfig {
  std::string model;
  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  SpeakerEmbeddingExtractorConfig() = default;
  SpeakerEmbeddingExtractorConfig(const std::string &model, int32_t num_threads,
                                  bool debug, const std::string &provider)
      : model(model),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

class SpeakerEmbeddingExtractorImpl;

class SpeakerEmbeddingExtractor {
 public:
  explicit SpeakerEmbeddingExtractor(
      const SpeakerEmbeddingExtractorConfig &config);

  template <typename Manager>
  SpeakerEmbeddingExtractor(Manager *mgr,
                            const SpeakerEmbeddingExtractorConfig &config);

  ~SpeakerEmbeddingExtractor();

  // Return the dimension of the embedding
  int32_t Dim() const;

  // Create a stream to accept audio samples and compute features
  std::unique_ptr<OnlineStream> CreateStream() const;

  // Return true if there are feature frames in OnlineStream that
  // can be used to compute embeddings.
  bool IsReady(OnlineStream *s) const;

  // Compute the speaker embedding from the available unprocessed features
  // of the given stream
  //
  // You have to ensure IsReady(s) returns true before you call this method.
  std::vector<float> Compute(OnlineStream *s) const;

 private:
  std::unique_ptr<SpeakerEmbeddingExtractorImpl> impl_;
};

}  // namespace sherpa_mnn

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_H_
