// sherpa-mnn/c-api/cxx-api.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-mnn/c-api/cxx-api.h"

#include <algorithm>
#include <cstring>

namespace sherpa_mnn::cxx {

Wave ReadWave(const std::string &filename) {
  auto p = SherpaMnnReadWave(filename.c_str());

  Wave ans;
  if (p) {
    ans.samples.resize(p->num_samples);

    std::copy(p->samples, p->samples + p->num_samples, ans.samples.data());

    ans.sample_rate = p->sample_rate;
    SherpaMnnFreeWave(p);
  }

  return ans;
}

bool WriteWave(const std::string &filename, const Wave &wave) {
  return SherpaMnnWriteWave(wave.samples.data(), wave.samples.size(),
                             wave.sample_rate, filename.c_str());
}

OnlineStream::OnlineStream(const SherpaMnnOnlineStream *p)
    : MoveOnly<OnlineStream, SherpaMnnOnlineStream>(p) {}

void OnlineStream::Destroy(const SherpaMnnOnlineStream *p) const {
  SherpaMnnDestroyOnlineStream(p);
}

void OnlineStream::AcceptWaveform(int32_t sample_rate, const float *samples,
                                  int32_t n) const {
  SherpaMnnOnlineStreamAcceptWaveform(p_, sample_rate, samples, n);
}

void OnlineStream::InputFinished() const {
  SherpaMnnOnlineStreamInputFinished(p_);
}

OnlineRecognizer OnlineRecognizer::Create(
    const OnlineRecognizerConfig &config) {
  struct SherpaMnnOnlineRecognizerConfig c;
  memset(&c, 0, sizeof(c));

  c.feat_config.sample_rate = config.feat_config.sample_rate;
  c.feat_config.feature_dim = config.feat_config.feature_dim;

  c.model_config.transducer.encoder =
      config.model_config.transducer.encoder.c_str();
  c.model_config.transducer.decoder =
      config.model_config.transducer.decoder.c_str();
  c.model_config.transducer.joiner =
      config.model_config.transducer.joiner.c_str();

  c.model_config.paraformer.encoder =
      config.model_config.paraformer.encoder.c_str();
  c.model_config.paraformer.decoder =
      config.model_config.paraformer.decoder.c_str();

  c.model_config.zipformer2_ctc.model =
      config.model_config.zipformer2_ctc.model.c_str();

  c.model_config.tokens = config.model_config.tokens.c_str();
  c.model_config.num_threads = config.model_config.num_threads;
  c.model_config.provider = config.model_config.provider.c_str();
  c.model_config.debug = config.model_config.debug;
  c.model_config.model_type = config.model_config.model_type.c_str();
  c.model_config.modeling_unit = config.model_config.modeling_unit.c_str();
  c.model_config.bpe_vocab = config.model_config.bpe_vocab.c_str();
  c.model_config.tokens_buf = config.model_config.tokens_buf.c_str();
  c.model_config.tokens_buf_size = config.model_config.tokens_buf.size();

  c.decoding_method = config.decoding_method.c_str();
  c.max_active_paths = config.max_active_paths;
  c.enable_endpoint = config.enable_endpoint;
  c.rule1_min_trailing_silence = config.rule1_min_trailing_silence;
  c.rule2_min_trailing_silence = config.rule2_min_trailing_silence;
  c.rule3_min_utterance_length = config.rule3_min_utterance_length;
  c.hotwords_file = config.hotwords_file.c_str();
  c.hotwords_score = config.hotwords_score;

  c.ctc_fst_decoder_config.graph = config.ctc_fst_decoder_config.graph.c_str();
  c.ctc_fst_decoder_config.max_active =
      config.ctc_fst_decoder_config.max_active;

  c.rule_fsts = config.rule_fsts.c_str();
  c.rule_fars = config.rule_fars.c_str();

  c.blank_penalty = config.blank_penalty;

  c.hotwords_buf = config.hotwords_buf.c_str();
  c.hotwords_buf_size = config.hotwords_buf.size();

  auto p = SherpaMnnCreateOnlineRecognizer(&c);
  return OnlineRecognizer(p);
}

OnlineRecognizer::OnlineRecognizer(const SherpaMnnOnlineRecognizer *p)
    : MoveOnly<OnlineRecognizer, SherpaMnnOnlineRecognizer>(p) {}

void OnlineRecognizer::Destroy(const SherpaMnnOnlineRecognizer *p) const {
  SherpaMnnDestroyOnlineRecognizer(p);
}

OnlineStream OnlineRecognizer::CreateStream() const {
  auto s = SherpaMnnCreateOnlineStream(p_);
  return OnlineStream{s};
}

OnlineStream OnlineRecognizer::CreateStream(const std::string &hotwords) const {
  auto s = SherpaMnnCreateOnlineStreamWithHotwords(p_, hotwords.c_str());
  return OnlineStream{s};
}

bool OnlineRecognizer::IsReady(const OnlineStream *s) const {
  return SherpaMnnIsOnlineStreamReady(p_, s->Get());
}

void OnlineRecognizer::Decode(const OnlineStream *s) const {
  SherpaMnnDecodeOnlineStream(p_, s->Get());
}

void OnlineRecognizer::Reset(const OnlineStream *s) const {
  SherpaMnnOnlineStreamReset(p_, s->Get());
}

bool OnlineRecognizer::IsEndpoint(const OnlineStream *s) const {
  return SherpaMnnOnlineStreamIsEndpoint(p_, s->Get());
}

void OnlineRecognizer::Decode(const OnlineStream *ss, int32_t n) const {
  if (n <= 0) {
    return;
  }

  std::vector<const SherpaMnnOnlineStream *> streams(n);
  for (int32_t i = 0; i != n; ++n) {
    streams[i] = ss[i].Get();
  }

  SherpaMnnDecodeMultipleOnlineStreams(p_, streams.data(), n);
}

OnlineRecognizerResult OnlineRecognizer::GetResult(
    const OnlineStream *s) const {
  auto r = SherpaMnnGetOnlineStreamResult(p_, s->Get());

  OnlineRecognizerResult ans;
  ans.text = r->text;

  ans.tokens.resize(r->count);
  for (int32_t i = 0; i != r->count; ++i) {
    ans.tokens[i] = r->tokens_arr[i];
  }

  if (r->timestamps) {
    ans.timestamps.resize(r->count);
    std::copy(r->timestamps, r->timestamps + r->count, ans.timestamps.data());
  }

  ans.json = r->json;

  SherpaMnnDestroyOnlineRecognizerResult(r);

  return ans;
}

// ============================================================================
// Non-streaming ASR
// ============================================================================
OfflineStream::OfflineStream(const SherpaMnnOfflineStream *p)
    : MoveOnly<OfflineStream, SherpaMnnOfflineStream>(p) {}

void OfflineStream::Destroy(const SherpaMnnOfflineStream *p) const {
  SherpaMnnDestroyOfflineStream(p);
}

void OfflineStream::AcceptWaveform(int32_t sample_rate, const float *samples,
                                   int32_t n) const {
  SherpaMnnAcceptWaveformOffline(p_, sample_rate, samples, n);
}

OfflineRecognizer OfflineRecognizer::Create(
    const OfflineRecognizerConfig &config) {
  struct SherpaMnnOfflineRecognizerConfig c;
  memset(&c, 0, sizeof(c));

  c.feat_config.sample_rate = config.feat_config.sample_rate;
  c.feat_config.feature_dim = config.feat_config.feature_dim;
  c.model_config.transducer.encoder =
      config.model_config.transducer.encoder.c_str();
  c.model_config.transducer.decoder =
      config.model_config.transducer.decoder.c_str();
  c.model_config.transducer.joiner =
      config.model_config.transducer.joiner.c_str();

  c.model_config.paraformer.model =
      config.model_config.paraformer.model.c_str();

  c.model_config.nemo_ctc.model = config.model_config.nemo_ctc.model.c_str();

  c.model_config.whisper.encoder = config.model_config.whisper.encoder.c_str();
  c.model_config.whisper.decoder = config.model_config.whisper.decoder.c_str();
  c.model_config.whisper.language =
      config.model_config.whisper.language.c_str();
  c.model_config.whisper.task = config.model_config.whisper.task.c_str();
  c.model_config.whisper.tail_paddings =
      config.model_config.whisper.tail_paddings;

  c.model_config.tdnn.model = config.model_config.tdnn.model.c_str();

  c.model_config.tokens = config.model_config.tokens.c_str();
  c.model_config.num_threads = config.model_config.num_threads;
  c.model_config.debug = config.model_config.debug;
  c.model_config.provider = config.model_config.provider.c_str();
  c.model_config.model_type = config.model_config.model_type.c_str();
  c.model_config.modeling_unit = config.model_config.modeling_unit.c_str();
  c.model_config.bpe_vocab = config.model_config.bpe_vocab.c_str();
  c.model_config.telespeech_ctc = config.model_config.telespeech_ctc.c_str();

  c.model_config.sense_voice.model =
      config.model_config.sense_voice.model.c_str();
  c.model_config.sense_voice.language =
      config.model_config.sense_voice.language.c_str();
  c.model_config.sense_voice.use_itn = config.model_config.sense_voice.use_itn;

  c.model_config.moonshine.preprocessor =
      config.model_config.moonshine.preprocessor.c_str();
  c.model_config.moonshine.encoder =
      config.model_config.moonshine.encoder.c_str();
  c.model_config.moonshine.uncached_decoder =
      config.model_config.moonshine.uncached_decoder.c_str();
  c.model_config.moonshine.cached_decoder =
      config.model_config.moonshine.cached_decoder.c_str();

  c.model_config.fire_red_asr.encoder =
      config.model_config.fire_red_asr.encoder.c_str();
  c.model_config.fire_red_asr.decoder =
      config.model_config.fire_red_asr.decoder.c_str();

  c.lm_config.model = config.lm_config.model.c_str();
  c.lm_config.scale = config.lm_config.scale;

  c.decoding_method = config.decoding_method.c_str();
  c.max_active_paths = config.max_active_paths;
  c.hotwords_file = config.hotwords_file.c_str();
  c.hotwords_score = config.hotwords_score;

  c.rule_fsts = config.rule_fsts.c_str();
  c.rule_fars = config.rule_fars.c_str();

  c.blank_penalty = config.blank_penalty;

  auto p = SherpaMnnCreateOfflineRecognizer(&c);
  return OfflineRecognizer(p);
}

OfflineRecognizer::OfflineRecognizer(const SherpaMnnOfflineRecognizer *p)
    : MoveOnly<OfflineRecognizer, SherpaMnnOfflineRecognizer>(p) {}

void OfflineRecognizer::Destroy(const SherpaMnnOfflineRecognizer *p) const {
  SherpaMnnDestroyOfflineRecognizer(p_);
}

OfflineStream OfflineRecognizer::CreateStream() const {
  auto s = SherpaMnnCreateOfflineStream(p_);
  return OfflineStream{s};
}

OfflineStream OfflineRecognizer::CreateStream(
    const std::string &hotwords) const {
  auto s = SherpaMnnCreateOfflineStreamWithHotwords(p_, hotwords.c_str());
  return OfflineStream{s};
}

void OfflineRecognizer::Decode(const OfflineStream *s) const {
  SherpaMnnDecodeOfflineStream(p_, s->Get());
}

void OfflineRecognizer::Decode(const OfflineStream *ss, int32_t n) const {
  if (n <= 0) {
    return;
  }

  std::vector<const SherpaMnnOfflineStream *> streams(n);
  for (int32_t i = 0; i != n; ++i) {
    streams[i] = ss[i].Get();
  }

  SherpaMnnDecodeMultipleOfflineStreams(p_, streams.data(), n);
}

OfflineRecognizerResult OfflineRecognizer::GetResult(
    const OfflineStream *s) const {
  auto r = SherpaMnnGetOfflineStreamResult(s->Get());

  OfflineRecognizerResult ans;
  if (r) {
    ans.text = r->text;

    if (r->timestamps) {
      ans.timestamps.resize(r->count);
      std::copy(r->timestamps, r->timestamps + r->count, ans.timestamps.data());
    }

    ans.tokens.resize(r->count);
    for (int32_t i = 0; i != r->count; ++i) {
      ans.tokens[i] = r->tokens_arr[i];
    }

    ans.json = r->json;
    ans.lang = r->lang ? r->lang : "";
    ans.emotion = r->emotion ? r->emotion : "";
    ans.event = r->event ? r->event : "";
  }

  SherpaMnnDestroyOfflineRecognizerResult(r);

  return ans;
}

OfflineTts OfflineTts::Create(const OfflineTtsConfig &config) {
  struct SherpaMnnOfflineTtsConfig c;
  memset(&c, 0, sizeof(c));

  c.model.vits.model = config.model.vits.model.c_str();
  c.model.vits.lexicon = config.model.vits.lexicon.c_str();
  c.model.vits.tokens = config.model.vits.tokens.c_str();
  c.model.vits.data_dir = config.model.vits.data_dir.c_str();
  c.model.vits.noise_scale = config.model.vits.noise_scale;
  c.model.vits.noise_scale_w = config.model.vits.noise_scale_w;
  c.model.vits.length_scale = config.model.vits.length_scale;
  c.model.vits.dict_dir = config.model.vits.dict_dir.c_str();

  c.model.matcha.acoustic_model = config.model.matcha.acoustic_model.c_str();
  c.model.matcha.vocoder = config.model.matcha.vocoder.c_str();
  c.model.matcha.lexicon = config.model.matcha.lexicon.c_str();
  c.model.matcha.tokens = config.model.matcha.tokens.c_str();
  c.model.matcha.data_dir = config.model.matcha.data_dir.c_str();
  c.model.matcha.noise_scale = config.model.matcha.noise_scale;
  c.model.matcha.length_scale = config.model.matcha.length_scale;
  c.model.matcha.dict_dir = config.model.matcha.dict_dir.c_str();

  c.model.kokoro.model = config.model.kokoro.model.c_str();
  c.model.kokoro.voices = config.model.kokoro.voices.c_str();
  c.model.kokoro.tokens = config.model.kokoro.tokens.c_str();
  c.model.kokoro.data_dir = config.model.kokoro.data_dir.c_str();
  c.model.kokoro.length_scale = config.model.kokoro.length_scale;
  c.model.kokoro.dict_dir = config.model.kokoro.dict_dir.c_str();
  c.model.kokoro.lexicon = config.model.kokoro.lexicon.c_str();

  c.model.num_threads = config.model.num_threads;
  c.model.debug = config.model.debug;
  c.model.provider = config.model.provider.c_str();

  c.rule_fsts = config.rule_fsts.c_str();
  c.max_num_sentences = config.max_num_sentences;
  c.silence_scale = config.silence_scale;
  c.rule_fars = config.rule_fars.c_str();

  auto p = SherpaMnnCreateOfflineTts(&c);
  return OfflineTts(p);
}

OfflineTts::OfflineTts(const SherpaMnnOfflineTts *p)
    : MoveOnly<OfflineTts, SherpaMnnOfflineTts>(p) {}

void OfflineTts::Destroy(const SherpaMnnOfflineTts *p) const {
  SherpaMnnDestroyOfflineTts(p);
}

int32_t OfflineTts::SampleRate() const {
  return SherpaMnnOfflineTtsSampleRate(p_);
}

int32_t OfflineTts::NumSpeakers() const {
  return SherpaMnnOfflineTtsNumSpeakers(p_);
}

GeneratedAudio OfflineTts::Generate(const std::string &text,
                                    int32_t sid /*= 0*/, float speed /*= 1.0*/,
                                    OfflineTtsCallback callback /*= nullptr*/,
                                    void *arg /*= nullptr*/) const {
  const SherpaMnnGeneratedAudio *audio;
  if (!callback) {
    audio = SherpaMnnOfflineTtsGenerate(p_, text.c_str(), sid, speed);
  } else {
    audio = SherpaMnnOfflineTtsGenerateWithProgressCallbackWithArg(
        p_, text.c_str(), sid, speed, callback, arg);
  }

  GeneratedAudio ans;
  ans.samples = std::vector<float>{audio->samples, audio->samples + audio->n};
  ans.sample_rate = audio->sample_rate;

  SherpaMnnDestroyOfflineTtsGeneratedAudio(audio);
  return ans;
}

KeywordSpotter KeywordSpotter::Create(const KeywordSpotterConfig &config) {
  struct SherpaMnnKeywordSpotterConfig c;
  memset(&c, 0, sizeof(c));

  c.feat_config.sample_rate = config.feat_config.sample_rate;

  c.model_config.transducer.encoder =
      config.model_config.transducer.encoder.c_str();
  c.model_config.transducer.decoder =
      config.model_config.transducer.decoder.c_str();
  c.model_config.transducer.joiner =
      config.model_config.transducer.joiner.c_str();
  c.feat_config.feature_dim = config.feat_config.feature_dim;

  c.model_config.paraformer.encoder =
      config.model_config.paraformer.encoder.c_str();
  c.model_config.paraformer.decoder =
      config.model_config.paraformer.decoder.c_str();

  c.model_config.zipformer2_ctc.model =
      config.model_config.zipformer2_ctc.model.c_str();

  c.model_config.tokens = config.model_config.tokens.c_str();
  c.model_config.num_threads = config.model_config.num_threads;
  c.model_config.provider = config.model_config.provider.c_str();
  c.model_config.debug = config.model_config.debug;
  c.model_config.model_type = config.model_config.model_type.c_str();
  c.model_config.modeling_unit = config.model_config.modeling_unit.c_str();
  c.model_config.bpe_vocab = config.model_config.bpe_vocab.c_str();
  c.model_config.tokens_buf = config.model_config.tokens_buf.c_str();
  c.model_config.tokens_buf_size = config.model_config.tokens_buf.size();

  c.max_active_paths = config.max_active_paths;
  c.num_trailing_blanks = config.num_trailing_blanks;
  c.keywords_score = config.keywords_score;
  c.keywords_threshold = config.keywords_threshold;
  c.keywords_file = config.keywords_file.c_str();

  auto p = SherpaMnnCreateKeywordSpotter(&c);
  return KeywordSpotter(p);
}

KeywordSpotter::KeywordSpotter(const SherpaMnnKeywordSpotter *p)
    : MoveOnly<KeywordSpotter, SherpaMnnKeywordSpotter>(p) {}

void KeywordSpotter::Destroy(const SherpaMnnKeywordSpotter *p) const {
  SherpaMnnDestroyKeywordSpotter(p);
}

OnlineStream KeywordSpotter::CreateStream() const {
  auto s = SherpaMnnCreateKeywordStream(p_);
  return OnlineStream{s};
}

OnlineStream KeywordSpotter::CreateStream(const std::string &keywords) const {
  auto s = SherpaMnnCreateKeywordStreamWithKeywords(p_, keywords.c_str());
  return OnlineStream{s};
}

bool KeywordSpotter::IsReady(const OnlineStream *s) const {
  return SherpaMnnIsKeywordStreamReady(p_, s->Get());
}

void KeywordSpotter::Decode(const OnlineStream *s) const {
  return SherpaMnnDecodeKeywordStream(p_, s->Get());
}

void KeywordSpotter::Decode(const OnlineStream *ss, int32_t n) const {
  if (n <= 0) {
    return;
  }

  std::vector<const SherpaMnnOnlineStream *> streams(n);
  for (int32_t i = 0; i != n; ++n) {
    streams[i] = ss[i].Get();
  }

  SherpaMnnDecodeMultipleKeywordStreams(p_, streams.data(), n);
}

KeywordResult KeywordSpotter::GetResult(const OnlineStream *s) const {
  auto r = SherpaMnnGetKeywordResult(p_, s->Get());

  KeywordResult ans;
  ans.keyword = r->keyword;

  ans.tokens.resize(r->count);
  for (int32_t i = 0; i < r->count; ++i) {
    ans.tokens[i] = r->tokens_arr[i];
  }

  if (r->timestamps) {
    ans.timestamps.resize(r->count);
    std::copy(r->timestamps, r->timestamps + r->count, ans.timestamps.data());
  }

  ans.start_time = r->start_time;
  ans.json = r->json;

  SherpaMnnDestroyKeywordResult(r);

  return ans;
}

void KeywordSpotter::Reset(const OnlineStream *s) const {
  SherpaMnnResetKeywordStream(p_, s->Get());
}

// ============================================================
// For Offline Speech Enhancement
// ============================================================

OfflineSpeechDenoiser OfflineSpeechDenoiser::Create(
    const OfflineSpeechDenoiserConfig &config) {
  struct SherpaMnnOfflineSpeechDenoiserConfig c;
  memset(&c, 0, sizeof(c));

  c.model.gtcrn.model = config.model.gtcrn.model.c_str();

  c.model.num_threads = config.model.num_threads;
  c.model.provider = config.model.provider.c_str();
  c.model.debug = config.model.debug;

  auto p = SherpaMnnCreateOfflineSpeechDenoiser(&c);

  return OfflineSpeechDenoiser(p);
}

void OfflineSpeechDenoiser::Destroy(
    const SherpaMnnOfflineSpeechDenoiser *p) const {
  SherpaMnnDestroyOfflineSpeechDenoiser(p);
}

OfflineSpeechDenoiser::OfflineSpeechDenoiser(
    const SherpaMnnOfflineSpeechDenoiser *p)
    : MoveOnly<OfflineSpeechDenoiser, SherpaMnnOfflineSpeechDenoiser>(p) {}

DenoisedAudio OfflineSpeechDenoiser::Run(const float *samples, int32_t n,
                                         int32_t sample_rate) const {
  auto audio = SherpaMnnOfflineSpeechDenoiserRun(p_, samples, n, sample_rate);

  DenoisedAudio ans;
  ans.samples = {audio->samples, audio->samples + audio->n};
  ans.sample_rate = audio->sample_rate;
  SherpaMnnDestroyDenoisedAudio(audio);

  return ans;
}

int32_t OfflineSpeechDenoiser::GetSampleRate() const {
  return SherpaMnnOfflineSpeechDenoiserGetSampleRate(p_);
}

}  // namespace sherpa_mnn::cxx
