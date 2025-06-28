// sherpa-mnn/python/csrc/sherpa-mnn.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-mnn/python/csrc/sherpa-mnn.h"

#include "sherpa-mnn/python/csrc/alsa.h"
#include "sherpa-mnn/python/csrc/audio-tagging.h"
#include "sherpa-mnn/python/csrc/circular-buffer.h"
#include "sherpa-mnn/python/csrc/display.h"
#include "sherpa-mnn/python/csrc/endpoint.h"
#include "sherpa-mnn/python/csrc/features.h"
#include "sherpa-mnn/python/csrc/keyword-spotter.h"
#include "sherpa-mnn/python/csrc/offline-ctc-fst-decoder-config.h"
#include "sherpa-mnn/python/csrc/offline-lm-config.h"
#include "sherpa-mnn/python/csrc/offline-model-config.h"
#include "sherpa-mnn/python/csrc/offline-punctuation.h"
#include "sherpa-mnn/python/csrc/offline-recognizer.h"
#include "sherpa-mnn/python/csrc/offline-speech-denoiser.h"
#include "sherpa-mnn/python/csrc/offline-stream.h"
#include "sherpa-mnn/python/csrc/online-ctc-fst-decoder-config.h"
#include "sherpa-mnn/python/csrc/online-lm-config.h"
#include "sherpa-mnn/python/csrc/online-model-config.h"
#include "sherpa-mnn/python/csrc/online-punctuation.h"
#include "sherpa-mnn/python/csrc/online-recognizer.h"
#include "sherpa-mnn/python/csrc/online-stream.h"
#include "sherpa-mnn/python/csrc/speaker-embedding-extractor.h"
#include "sherpa-mnn/python/csrc/speaker-embedding-manager.h"
#include "sherpa-mnn/python/csrc/spoken-language-identification.h"
#include "sherpa-mnn/python/csrc/vad-model-config.h"
#include "sherpa-mnn/python/csrc/vad-model.h"
#include "sherpa-mnn/python/csrc/voice-activity-detector.h"
#include "sherpa-mnn/python/csrc/wave-writer.h"

#if SHERPA_MNN_ENABLE_TTS == 1
#include "sherpa-mnn/python/csrc/offline-tts.h"
#endif

#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1
#include "sherpa-mnn/python/csrc/fast-clustering.h"
#include "sherpa-mnn/python/csrc/offline-speaker-diarization-result.h"
#include "sherpa-mnn/python/csrc/offline-speaker-diarization.h"
#endif

namespace sherpa_mnn {

PYBIND11_MODULE(_sherpa_mnn, m) {
  m.doc() = "pybind11 binding of sherpa-mnn";

  PybindWaveWriter(&m);
  PybindAudioTagging(&m);
  PybindOfflinePunctuation(&m);
  PybindOnlinePunctuation(&m);

  PybindFeatures(&m);
  PybindOnlineCtcFstDecoderConfig(&m);
  PybindOnlineModelConfig(&m);
  PybindOnlineLMConfig(&m);
  PybindOnlineStream(&m);
  PybindEndpoint(&m);
  PybindOnlineRecognizer(&m);
  PybindKeywordSpotter(&m);
  PybindDisplay(&m);

  PybindOfflineStream(&m);
  PybindOfflineLMConfig(&m);
  PybindOfflineModelConfig(&m);
  PybindOfflineCtcFstDecoderConfig(&m);
  PybindOfflineRecognizer(&m);

  PybindVadModelConfig(&m);
  PybindVadModel(&m);
  PybindCircularBuffer(&m);
  PybindVoiceActivityDetector(&m);

#if SHERPA_MNN_ENABLE_TTS == 1
  PybindOfflineTts(&m);
#endif

  PybindSpeakerEmbeddingExtractor(&m);
  PybindSpeakerEmbeddingManager(&m);
  PybindSpokenLanguageIdentification(&m);

#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1
  PybindFastClustering(&m);
  PybindOfflineSpeakerDiarizationResult(&m);
  PybindOfflineSpeakerDiarization(&m);
#endif

  PybindAlsa(&m);
  PybindOfflineSpeechDenoiser(&m);
}

}  // namespace sherpa_mnn
