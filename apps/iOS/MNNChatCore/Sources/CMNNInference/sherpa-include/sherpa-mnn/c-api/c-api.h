// sherpa-mnn/c-api/c-api.h
//
// Copyright (c)  2023  Xiaomi Corporation

// C API for sherpa-mnn
//
// Please refer to
// https://github.com/k2-fsa/sherpa-mnn/blob/master/c-api-examples/decode-file-c-api.c
// for usages.
//

#ifndef SHERPA_ONNX_C_API_C_API_H_
#define SHERPA_ONNX_C_API_C_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// See https://github.com/pytorch/pytorch/blob/main/c10/macros/Export.h
// We will set SHERPA_ONNX_BUILD_SHARED_LIBS and SHERPA_ONNX_BUILD_MAIN_LIB in
// CMakeLists.txt

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#if defined(_WIN32)
#if defined(SHERPA_ONNX_BUILD_SHARED_LIBS)
#define SHERPA_ONNX_EXPORT __declspec(dllexport)
#define SHERPA_ONNX_IMPORT __declspec(dllimport)
#else
#define SHERPA_ONNX_EXPORT
#define SHERPA_ONNX_IMPORT
#endif
#else  // WIN32
#define SHERPA_ONNX_EXPORT __attribute__((visibility("default")))

#define SHERPA_ONNX_IMPORT SHERPA_ONNX_EXPORT
#endif  // WIN32

#if defined(SHERPA_ONNX_BUILD_MAIN_LIB)
#define SHERPA_ONNX_API SHERPA_ONNX_EXPORT
#else
#define SHERPA_ONNX_API SHERPA_ONNX_IMPORT
#endif

/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to download pre-trained models. That is, you can find encoder-xxx.onnx
/// decoder-xxx.onnx, joiner-xxx.onnx, and tokens.txt for this struct
/// from there.
SHERPA_ONNX_API typedef struct SherpaMnnOnlineTransducerModelConfig {
  const char *encoder;
  const char *decoder;
  const char *joiner;
} SherpaMnnOnlineTransducerModelConfig;

// please visit
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/index.html
// to download pre-trained streaming paraformer models
SHERPA_ONNX_API typedef struct SherpaMnnOnlineParaformerModelConfig {
  const char *encoder;
  const char *decoder;
} SherpaMnnOnlineParaformerModelConfig;

// Please visit
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/zipformer-ctc-models.html#
// to download pre-trained streaming zipformer2 ctc models
SHERPA_ONNX_API typedef struct SherpaMnnOnlineZipformer2CtcModelConfig {
  const char *model;
} SherpaMnnOnlineZipformer2CtcModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOnlineModelConfig {
  SherpaMnnOnlineTransducerModelConfig transducer;
  SherpaMnnOnlineParaformerModelConfig paraformer;
  SherpaMnnOnlineZipformer2CtcModelConfig zipformer2_ctc;
  const char *tokens;
  int32_t num_threads;
  const char *provider;
  int32_t debug;  // true to print debug information of the model
  const char *model_type;
  // Valid values:
  //  - cjkchar
  //  - bpe
  //  - cjkchar+bpe
  const char *modeling_unit;
  const char *bpe_vocab;
  /// if non-null, loading the tokens from the buffer instead of from the
  /// "tokens" file
  const char *tokens_buf;
  /// byte size excluding the trailing '\0'
  int32_t tokens_buf_size;
} SherpaMnnOnlineModelConfig;

/// It expects 16 kHz 16-bit single channel wave format.
SHERPA_ONNX_API typedef struct SherpaMnnFeatureConfig {
  /// Sample rate of the input data. MUST match the one expected
  /// by the model. For instance, it should be 16000 for models provided
  /// by us.
  int32_t sample_rate;

  /// Feature dimension of the model.
  /// For instance, it should be 80 for models provided by us.
  int32_t feature_dim;
} SherpaMnnFeatureConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOnlineCtcFstDecoderConfig {
  const char *graph;
  int32_t max_active;
} SherpaMnnOnlineCtcFstDecoderConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOnlineRecognizerConfig {
  SherpaMnnFeatureConfig feat_config;
  SherpaMnnOnlineModelConfig model_config;

  /// Possible values are: greedy_search, modified_beam_search
  const char *decoding_method;

  /// Used only when decoding_method is modified_beam_search
  /// Example value: 4
  int32_t max_active_paths;

  /// 0 to disable endpoint detection.
  /// A non-zero value to enable endpoint detection.
  int32_t enable_endpoint;

  /// An endpoint is detected if trailing silence in seconds is larger than
  /// this value even if nothing has been decoded.
  /// Used only when enable_endpoint is not 0.
  float rule1_min_trailing_silence;

  /// An endpoint is detected if trailing silence in seconds is larger than
  /// this value after something that is not blank has been decoded.
  /// Used only when enable_endpoint is not 0.
  float rule2_min_trailing_silence;

  /// An endpoint is detected if the utterance in seconds is larger than
  /// this value.
  /// Used only when enable_endpoint is not 0.
  float rule3_min_utterance_length;

  /// Path to the hotwords.
  const char *hotwords_file;

  /// Bonus score for each token in hotwords.
  float hotwords_score;

  SherpaMnnOnlineCtcFstDecoderConfig ctc_fst_decoder_config;
  const char *rule_fsts;
  const char *rule_fars;
  float blank_penalty;

  /// if non-nullptr, loading the hotwords from the buffered string directly in
  const char *hotwords_buf;
  /// byte size excluding the tailing '\0'
  int32_t hotwords_buf_size;
} SherpaMnnOnlineRecognizerConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOnlineRecognizerResult {
  // Recognized text
  const char *text;

  // Pointer to continuous memory which holds string based tokens
  // which are separated by \0
  const char *tokens;

  // a pointer array containing the address of the first item in tokens
  const char *const *tokens_arr;

  // Pointer to continuous memory which holds timestamps
  //
  // Caution: If timestamp information is not available, this pointer is NULL.
  // Please check whether it is NULL before you access it; otherwise, you would
  // get segmentation fault.
  float *timestamps;

  // The number of tokens/timestamps in above pointer
  int32_t count;

  /** Return a json string.
   *
   * The returned string contains:
   *   {
   *     "text": "The recognition result",
   *     "tokens": [x, x, x],
   *     "timestamps": [x, x, x],
   *     "segment": x,
   *     "start_time": x,
   *     "is_final": true|false
   *   }
   */
  const char *json;
} SherpaMnnOnlineRecognizerResult;

/// Note: OnlineRecognizer here means StreamingRecognizer.
/// It does not need to access the Internet during recognition.
/// Everything is run locally.
SHERPA_ONNX_API typedef struct SherpaMnnOnlineRecognizer
    SherpaMnnOnlineRecognizer;
SHERPA_ONNX_API typedef struct SherpaMnnOnlineStream SherpaMnnOnlineStream;

/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          SherpaMnnDestroyOnlineRecognizer() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOnlineRecognizer *
SherpaMnnCreateOnlineRecognizer(
    const SherpaMnnOnlineRecognizerConfig *config);

/// Free a pointer returned by SherpaMnnCreateOnlineRecognizer()
///
/// @param p A pointer returned by SherpaMnnCreateOnlineRecognizer()
SHERPA_ONNX_API void SherpaMnnDestroyOnlineRecognizer(
    const SherpaMnnOnlineRecognizer *recognizer);

/// Create an online stream for accepting wave samples.
///
/// @param recognizer  A pointer returned by SherpaMnnCreateOnlineRecognizer()
/// @return Return a pointer to an OnlineStream. The user has to invoke
///         SherpaMnnDestroyOnlineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOnlineStream *SherpaMnnCreateOnlineStream(
    const SherpaMnnOnlineRecognizer *recognizer);

/// Create an online stream for accepting wave samples with the specified hot
/// words.
///
/// @param recognizer  A pointer returned by SherpaMnnCreateOnlineRecognizer()
/// @return Return a pointer to an OnlineStream. The user has to invoke
///         SherpaMnnDestroyOnlineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOnlineStream *
SherpaMnnCreateOnlineStreamWithHotwords(
    const SherpaMnnOnlineRecognizer *recognizer, const char *hotwords);

/// Destroy an online stream.
///
/// @param stream A pointer returned by SherpaMnnCreateOnlineStream()
SHERPA_ONNX_API void SherpaMnnDestroyOnlineStream(
    const SherpaMnnOnlineStream *stream);

/// Accept input audio samples and compute the features.
/// The user has to invoke SherpaMnnDecodeOnlineStream() to run the neural
/// network and decoding.
///
/// @param stream  A pointer returned by SherpaMnnCreateOnlineStream().
/// @param sample_rate  Sample rate of the input samples. If it is different
///                     from config.feat_config.sample_rate, we will do
///                     resampling inside sherpa-mnn.
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n  Number of elements in the samples array.
SHERPA_ONNX_API void SherpaMnnOnlineStreamAcceptWaveform(
    const SherpaMnnOnlineStream *stream, int32_t sample_rate,
    const float *samples, int32_t n);

/// Return 1 if there are enough number of feature frames for decoding.
/// Return 0 otherwise.
///
/// @param recognizer  A pointer returned by SherpaMnnCreateOnlineRecognizer
/// @param stream  A pointer returned by SherpaMnnCreateOnlineStream
SHERPA_ONNX_API int32_t
SherpaMnnIsOnlineStreamReady(const SherpaMnnOnlineRecognizer *recognizer,
                              const SherpaMnnOnlineStream *stream);

/// Call this function to run the neural network model and decoding.
//
/// Precondition for this function: SherpaMnnIsOnlineStreamReady() MUST
/// return 1.
///
/// Usage example:
///
///  while (SherpaMnnIsOnlineStreamReady(recognizer, stream)) {
///     SherpaMnnDecodeOnlineStream(recognizer, stream);
///  }
///
SHERPA_ONNX_API void SherpaMnnDecodeOnlineStream(
    const SherpaMnnOnlineRecognizer *recognizer,
    const SherpaMnnOnlineStream *stream);

/// This function is similar to SherpaMnnDecodeOnlineStream(). It decodes
/// multiple OnlineStream in parallel.
///
/// Caution: The caller has to ensure each OnlineStream is ready, i.e.,
/// SherpaMnnIsOnlineStreamReady() for that stream should return 1.
///
/// @param recognizer  A pointer returned by SherpaMnnCreateOnlineRecognizer()
/// @param streams  A pointer array containing pointers returned by
///                 SherpaMnnCreateOnlineRecognizer()
/// @param n  Number of elements in the given streams array.
SHERPA_ONNX_API void SherpaMnnDecodeMultipleOnlineStreams(
    const SherpaMnnOnlineRecognizer *recognizer,
    const SherpaMnnOnlineStream **streams, int32_t n);

/// Get the decoding results so far for an OnlineStream.
///
/// @param recognizer A pointer returned by SherpaMnnCreateOnlineRecognizer().
/// @param stream A pointer returned by SherpaMnnCreateOnlineStream().
/// @return A pointer containing the result. The user has to invoke
///         SherpaMnnDestroyOnlineRecognizerResult() to free the returned
///         pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOnlineRecognizerResult *
SherpaMnnGetOnlineStreamResult(const SherpaMnnOnlineRecognizer *recognizer,
                                const SherpaMnnOnlineStream *stream);

/// Destroy the pointer returned by SherpaMnnGetOnlineStreamResult().
///
/// @param r A pointer returned by SherpaMnnGetOnlineStreamResult()
SHERPA_ONNX_API void SherpaMnnDestroyOnlineRecognizerResult(
    const SherpaMnnOnlineRecognizerResult *r);

/// Return the result as a json string.
/// The user has to invoke
/// SherpaMnnDestroyOnlineStreamResultJson()
/// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const char *SherpaMnnGetOnlineStreamResultAsJson(
    const SherpaMnnOnlineRecognizer *recognizer,
    const SherpaMnnOnlineStream *stream);

SHERPA_ONNX_API void SherpaMnnDestroyOnlineStreamResultJson(const char *s);

/// SherpaMnnOnlineStreamReset an OnlineStream , which clears the neural
/// network model state and the state for decoding.
///
/// @param recognizer A pointer returned by SherpaMnnCreateOnlineRecognizer().
/// @param stream A pointer returned by SherpaMnnCreateOnlineStream
SHERPA_ONNX_API void SherpaMnnOnlineStreamReset(
    const SherpaMnnOnlineRecognizer *recognizer,
    const SherpaMnnOnlineStream *stream);

/// Signal that no more audio samples would be available.
/// After this call, you cannot call SherpaMnnOnlineStreamAcceptWaveform() any
/// more.
///
/// @param stream A pointer returned by SherpaMnnCreateOnlineStream()
SHERPA_ONNX_API void SherpaMnnOnlineStreamInputFinished(
    const SherpaMnnOnlineStream *stream);

/// Return 1 if an endpoint has been detected.
///
/// @param recognizer A pointer returned by SherpaMnnCreateOnlineRecognizer()
/// @param stream A pointer returned by SherpaMnnCreateOnlineStream()
/// @return Return 1 if an endpoint is detected. Return 0 otherwise.
SHERPA_ONNX_API int32_t
SherpaMnnOnlineStreamIsEndpoint(const SherpaMnnOnlineRecognizer *recognizer,
                                 const SherpaMnnOnlineStream *stream);

// for displaying results on Linux/macOS.
SHERPA_ONNX_API typedef struct SherpaMnnDisplay SherpaMnnDisplay;

/// Create a display object. Must be freed using SherpaMnnDestroyDisplay to
/// avoid memory leak.
SHERPA_ONNX_API const SherpaMnnDisplay *SherpaMnnCreateDisplay(
    int32_t max_word_per_line);

SHERPA_ONNX_API void SherpaMnnDestroyDisplay(const SherpaMnnDisplay *display);

/// Print the result.
SHERPA_ONNX_API void SherpaMnnPrint(const SherpaMnnDisplay *display,
                                     int32_t idx, const char *s);
// ============================================================
// For offline ASR (i.e., non-streaming ASR)
// ============================================================

/// Please refer to
/// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
/// to download pre-trained models. That is, you can find encoder-xxx.onnx
/// decoder-xxx.onnx, and joiner-xxx.onnx for this struct
/// from there.
SHERPA_ONNX_API typedef struct SherpaMnnOfflineTransducerModelConfig {
  const char *encoder;
  const char *decoder;
  const char *joiner;
} SherpaMnnOfflineTransducerModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineParaformerModelConfig {
  const char *model;
} SherpaMnnOfflineParaformerModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineNemoEncDecCtcModelConfig {
  const char *model;
} SherpaMnnOfflineNemoEncDecCtcModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineWhisperModelConfig {
  const char *encoder;
  const char *decoder;
  const char *language;
  const char *task;
  int32_t tail_paddings;
} SherpaMnnOfflineWhisperModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineFireRedAsrModelConfig {
  const char *encoder;
  const char *decoder;
} SherpaMnnOfflineFireRedAsrModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineMoonshineModelConfig {
  const char *preprocessor;
  const char *encoder;
  const char *uncached_decoder;
  const char *cached_decoder;
} SherpaMnnOfflineMoonshineModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineTdnnModelConfig {
  const char *model;
} SherpaMnnOfflineTdnnModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineLMConfig {
  const char *model;
  float scale;
} SherpaMnnOfflineLMConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineSenseVoiceModelConfig {
  const char *model;
  const char *language;
  int32_t use_itn;
} SherpaMnnOfflineSenseVoiceModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineModelConfig {
  SherpaMnnOfflineTransducerModelConfig transducer;
  SherpaMnnOfflineParaformerModelConfig paraformer;
  SherpaMnnOfflineNemoEncDecCtcModelConfig nemo_ctc;
  SherpaMnnOfflineWhisperModelConfig whisper;
  SherpaMnnOfflineTdnnModelConfig tdnn;

  const char *tokens;
  int32_t num_threads;
  int32_t debug;
  const char *provider;
  const char *model_type;
  // Valid values:
  //  - cjkchar
  //  - bpe
  //  - cjkchar+bpe
  const char *modeling_unit;
  const char *bpe_vocab;
  const char *telespeech_ctc;
  SherpaMnnOfflineSenseVoiceModelConfig sense_voice;
  SherpaMnnOfflineMoonshineModelConfig moonshine;
  SherpaMnnOfflineFireRedAsrModelConfig fire_red_asr;
} SherpaMnnOfflineModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineRecognizerConfig {
  SherpaMnnFeatureConfig feat_config;
  SherpaMnnOfflineModelConfig model_config;
  SherpaMnnOfflineLMConfig lm_config;

  const char *decoding_method;
  int32_t max_active_paths;

  /// Path to the hotwords.
  const char *hotwords_file;

  /// Bonus score for each token in hotwords.
  float hotwords_score;
  const char *rule_fsts;
  const char *rule_fars;
  float blank_penalty;
} SherpaMnnOfflineRecognizerConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineRecognizer
    SherpaMnnOfflineRecognizer;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineStream SherpaMnnOfflineStream;

/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          SherpaMnnDestroyOfflineRecognizer() to free it to avoid memory
//          leak.
SHERPA_ONNX_API const SherpaMnnOfflineRecognizer *
SherpaMnnCreateOfflineRecognizer(
    const SherpaMnnOfflineRecognizerConfig *config);

/// @param config  Config for the recognizer.
SHERPA_ONNX_API void SherpaMnnOfflineRecognizerSetConfig(
    const SherpaMnnOfflineRecognizer *recognizer,
    const SherpaMnnOfflineRecognizerConfig *config);

/// Free a pointer returned by SherpaMnnCreateOfflineRecognizer()
///
/// @param p A pointer returned by SherpaMnnCreateOfflineRecognizer()
SHERPA_ONNX_API void SherpaMnnDestroyOfflineRecognizer(
    const SherpaMnnOfflineRecognizer *recognizer);

/// Create an offline stream for accepting wave samples.
///
/// @param recognizer  A pointer returned by SherpaMnnCreateOfflineRecognizer()
/// @return Return a pointer to an OfflineStream. The user has to invoke
///         SherpaMnnDestroyOfflineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOfflineStream *SherpaMnnCreateOfflineStream(
    const SherpaMnnOfflineRecognizer *recognizer);

/// Create an offline stream for accepting wave samples with the specified hot
/// words.
///
/// @param recognizer  A pointer returned by SherpaMnnCreateOfflineRecognizer()
/// @return Return a pointer to an OfflineStream. The user has to invoke
///         SherpaMnnDestroyOfflineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOfflineStream *
SherpaMnnCreateOfflineStreamWithHotwords(
    const SherpaMnnOfflineRecognizer *recognizer, const char *hotwords);

/// Destroy an offline stream.
///
/// @param stream A pointer returned by SherpaMnnCreateOfflineStream()
SHERPA_ONNX_API void SherpaMnnDestroyOfflineStream(
    const SherpaMnnOfflineStream *stream);

/// Accept input audio samples and compute the features.
/// The user has to invoke SherpaMnnDecodeOfflineStream() to run the neural
/// network and decoding.
///
/// @param stream  A pointer returned by SherpaMnnCreateOfflineStream().
/// @param sample_rate  Sample rate of the input samples. If it is different
///                     from config.feat_config.sample_rate, we will do
///                     resampling inside sherpa-mnn.
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n  Number of elements in the samples array.
///
/// @caution: For each offline stream, please invoke this function only once!
SHERPA_ONNX_API void SherpaMnnAcceptWaveformOffline(
    const SherpaMnnOfflineStream *stream, int32_t sample_rate,
    const float *samples, int32_t n);
/// Decode an offline stream.
///
/// We assume you have invoked SherpaMnnAcceptWaveformOffline() for the given
/// stream before calling this function.
///
/// @param recognizer A pointer returned by SherpaMnnCreateOfflineRecognizer().
/// @param stream A pointer returned by SherpaMnnCreateOfflineStream()
SHERPA_ONNX_API void SherpaMnnDecodeOfflineStream(
    const SherpaMnnOfflineRecognizer *recognizer,
    const SherpaMnnOfflineStream *stream);

/// Decode a list offline streams in parallel.
///
/// We assume you have invoked SherpaMnnAcceptWaveformOffline() for each stream
/// before calling this function.
///
/// @param recognizer A pointer returned by SherpaMnnCreateOfflineRecognizer().
/// @param streams A pointer pointer array containing pointers returned
///                by SherpaMnnCreateOfflineStream().
/// @param n Number of entries in the given streams.
SHERPA_ONNX_API void SherpaMnnDecodeMultipleOfflineStreams(
    const SherpaMnnOfflineRecognizer *recognizer,
    const SherpaMnnOfflineStream **streams, int32_t n);

SHERPA_ONNX_API typedef struct SherpaMnnOfflineRecognizerResult {
  const char *text;

  // Pointer to continuous memory which holds timestamps
  //
  // It is NULL if the model does not support timestamps
  float *timestamps;

  // number of entries in timestamps
  int32_t count;

  // Pointer to continuous memory which holds string based tokens
  // which are separated by \0
  const char *tokens;

  // a pointer array containing the address of the first item in tokens
  const char *const *tokens_arr;

  /** Return a json string.
   *
   * The returned string contains:
   *   {
   *     "text": "The recognition result",
   *     "tokens": [x, x, x],
   *     "timestamps": [x, x, x],
   *     "segment": x,
   *     "start_time": x,
   *     "is_final": true|false
   *   }
   */
  const char *json;

  // return recognized language
  const char *lang;

  // return emotion.
  const char *emotion;

  // return event.
  const char *event;
} SherpaMnnOfflineRecognizerResult;

/// Get the result of the offline stream.
///
/// We assume you have called SherpaMnnDecodeOfflineStream() or
/// SherpaMnnDecodeMultipleOfflineStreams() with the given stream before
/// calling this function.
///
/// @param stream A pointer returned by SherpaMnnCreateOfflineStream().
/// @return Return a pointer to the result. The user has to invoke
///         SherpaMnnDestroyOnlineRecognizerResult() to free the returned
///         pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOfflineRecognizerResult *
SherpaMnnGetOfflineStreamResult(const SherpaMnnOfflineStream *stream);

/// Destroy the pointer returned by SherpaMnnGetOfflineStreamResult().
///
/// @param r A pointer returned by SherpaMnnGetOfflineStreamResult()
SHERPA_ONNX_API void SherpaMnnDestroyOfflineRecognizerResult(
    const SherpaMnnOfflineRecognizerResult *r);

/// Return the result as a json string.
/// The user has to use SherpaMnnDestroyOfflineStreamResultJson()
/// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const char *SherpaMnnGetOfflineStreamResultAsJson(
    const SherpaMnnOfflineStream *stream);

SHERPA_ONNX_API void SherpaMnnDestroyOfflineStreamResultJson(const char *s);

// ============================================================
// For Keyword Spotter
// ============================================================
SHERPA_ONNX_API typedef struct SherpaMnnKeywordResult {
  /// The triggered keyword.
  /// For English, it consists of space separated words.
  /// For Chinese, it consists of Chinese words without spaces.
  /// Example 1: "hello world"
  /// Example 2: "你好世界"
  const char *keyword;

  /// Decoded results at the token level.
  /// For instance, for BPE-based models it consists of a list of BPE tokens.
  const char *tokens;

  const char *const *tokens_arr;

  int32_t count;

  /// timestamps.size() == tokens.size()
  /// timestamps[i] records the time in seconds when tokens[i] is decoded.
  float *timestamps;

  /// Starting time of this segment.
  /// When an endpoint is detected, it will change
  float start_time;

  /** Return a json string.
   *
   * The returned string contains:
   *   {
   *     "keyword": "The triggered keyword",
   *     "tokens": [x, x, x],
   *     "timestamps": [x, x, x],
   *     "start_time": x,
   *   }
   */
  const char *json;
} SherpaMnnKeywordResult;

SHERPA_ONNX_API typedef struct SherpaMnnKeywordSpotterConfig {
  SherpaMnnFeatureConfig feat_config;
  SherpaMnnOnlineModelConfig model_config;
  int32_t max_active_paths;
  int32_t num_trailing_blanks;
  float keywords_score;
  float keywords_threshold;
  const char *keywords_file;
  /// if non-null, loading the keywords from the buffer instead of from the
  /// keywords_file
  const char *keywords_buf;
  /// byte size excluding the trailing '\0'
  int32_t keywords_buf_size;
} SherpaMnnKeywordSpotterConfig;

SHERPA_ONNX_API typedef struct SherpaMnnKeywordSpotter
    SherpaMnnKeywordSpotter;

/// @param config  Config for the keyword spotter.
/// @return Return a pointer to the spotter. The user has to invoke
///         SherpaMnnDestroyKeywordSpotter() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnKeywordSpotter *SherpaMnnCreateKeywordSpotter(
    const SherpaMnnKeywordSpotterConfig *config);

/// Free a pointer returned by SherpaMnnCreateKeywordSpotter()
///
/// @param p A pointer returned by SherpaMnnCreateKeywordSpotter()
SHERPA_ONNX_API void SherpaMnnDestroyKeywordSpotter(
    const SherpaMnnKeywordSpotter *spotter);

/// Create an online stream for accepting wave samples.
///
/// @param spotter A pointer returned by SherpaMnnCreateKeywordSpotter()
/// @return Return a pointer to an OnlineStream. The user has to invoke
///         SherpaMnnDestroyOnlineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOnlineStream *SherpaMnnCreateKeywordStream(
    const SherpaMnnKeywordSpotter *spotter);

/// Create an online stream for accepting wave samples with the specified hot
/// words.
///
/// @param spotter A pointer returned by SherpaMnnCreateKeywordSpotter()
/// @param keywords A pointer points to the keywords that you set
/// @return Return a pointer to an OnlineStream. The user has to invoke
///         SherpaMnnDestroyOnlineStream() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOnlineStream *
SherpaMnnCreateKeywordStreamWithKeywords(
    const SherpaMnnKeywordSpotter *spotter, const char *keywords);

/// Return 1 if there are enough number of feature frames for decoding.
/// Return 0 otherwise.
///
/// @param spotter A pointer returned by SherpaMnnCreateKeywordSpotter
/// @param stream  A pointer returned by SherpaMnnCreateKeywordStream
SHERPA_ONNX_API int32_t
SherpaMnnIsKeywordStreamReady(const SherpaMnnKeywordSpotter *spotter,
                               const SherpaMnnOnlineStream *stream);

/// Call this function to run the neural network model and decoding.
//
/// Precondition for this function: SherpaMnnIsKeywordStreamReady() MUST
/// return 1.
SHERPA_ONNX_API void SherpaMnnDecodeKeywordStream(
    const SherpaMnnKeywordSpotter *spotter,
    const SherpaMnnOnlineStream *stream);

/// Please call it right after a keyword is detected
SHERPA_ONNX_API void SherpaMnnResetKeywordStream(
    const SherpaMnnKeywordSpotter *spotter,
    const SherpaMnnOnlineStream *stream);

/// This function is similar to SherpaMnnDecodeKeywordStream(). It decodes
/// multiple OnlineStream in parallel.
///
/// Caution: The caller has to ensure each OnlineStream is ready, i.e.,
/// SherpaMnnIsKeywordStreamReady() for that stream should return 1.
///
/// @param spotter A pointer returned by SherpaMnnCreateKeywordSpotter()
/// @param streams  A pointer array containing pointers returned by
///                 SherpaMnnCreateKeywordStream()
/// @param n  Number of elements in the given streams array.
SHERPA_ONNX_API void SherpaMnnDecodeMultipleKeywordStreams(
    const SherpaMnnKeywordSpotter *spotter,
    const SherpaMnnOnlineStream **streams, int32_t n);

/// Get the decoding results so far for an OnlineStream.
///
/// @param spotter A pointer returned by SherpaMnnCreateKeywordSpotter().
/// @param stream A pointer returned by SherpaMnnCreateKeywordStream().
/// @return A pointer containing the result. The user has to invoke
///         SherpaMnnDestroyKeywordResult() to free the returned pointer to
///         avoid memory leak.
SHERPA_ONNX_API const SherpaMnnKeywordResult *SherpaMnnGetKeywordResult(
    const SherpaMnnKeywordSpotter *spotter,
    const SherpaMnnOnlineStream *stream);

/// Destroy the pointer returned by SherpaMnnGetKeywordResult().
///
/// @param r A pointer returned by SherpaMnnGetKeywordResult()
SHERPA_ONNX_API void SherpaMnnDestroyKeywordResult(
    const SherpaMnnKeywordResult *r);

// the user has to call SherpaMnnFreeKeywordResultJson() to free the returned
// pointer to avoid memory leak
SHERPA_ONNX_API const char *SherpaMnnGetKeywordResultAsJson(
    const SherpaMnnKeywordSpotter *spotter,
    const SherpaMnnOnlineStream *stream);

SHERPA_ONNX_API void SherpaMnnFreeKeywordResultJson(const char *s);

// ============================================================
// For VAD
// ============================================================

SHERPA_ONNX_API typedef struct SherpaMnnSileroVadModelConfig {
  // Path to the silero VAD model
  const char *model;

  // threshold to classify a segment as speech
  //
  // If the predicted probability of a segment is larger than this
  // value, then it is classified as speech.
  float threshold;

  // in seconds
  float min_silence_duration;

  // in seconds
  float min_speech_duration;

  int window_size;

  // If a speech segment is longer than this value, then we increase
  // the threshold to 0.9. After finishing detecting the segment,
  // the threshold value is reset to its original value.
  float max_speech_duration;
} SherpaMnnSileroVadModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnVadModelConfig {
  SherpaMnnSileroVadModelConfig silero_vad;

  int32_t sample_rate;
  int32_t num_threads;
  const char *provider;
  int32_t debug;
} SherpaMnnVadModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnCircularBuffer
    SherpaMnnCircularBuffer;

// Return an instance of circular buffer. The user has to use
// SherpaMnnDestroyCircularBuffer() to free the returned pointer to avoid
// memory leak.
SHERPA_ONNX_API const SherpaMnnCircularBuffer *SherpaMnnCreateCircularBuffer(
    int32_t capacity);

// Free the pointer returned by SherpaMnnCreateCircularBuffer()
SHERPA_ONNX_API void SherpaMnnDestroyCircularBuffer(
    const SherpaMnnCircularBuffer *buffer);

SHERPA_ONNX_API void SherpaMnnCircularBufferPush(
    const SherpaMnnCircularBuffer *buffer, const float *p, int32_t n);

// Return n samples starting at the given index.
//
// Return a pointer to an array containing n samples starting at start_index.
// The user has to use SherpaMnnCircularBufferFree() to free the returned
// pointer to avoid memory leak.
SHERPA_ONNX_API const float *SherpaMnnCircularBufferGet(
    const SherpaMnnCircularBuffer *buffer, int32_t start_index, int32_t n);

// Free the pointer returned by SherpaMnnCircularBufferGet().
SHERPA_ONNX_API void SherpaMnnCircularBufferFree(const float *p);

// Remove n elements from the buffer
SHERPA_ONNX_API void SherpaMnnCircularBufferPop(
    const SherpaMnnCircularBuffer *buffer, int32_t n);

// Return number of elements in the buffer.
SHERPA_ONNX_API int32_t
SherpaMnnCircularBufferSize(const SherpaMnnCircularBuffer *buffer);

// Return the head of the buffer. It's always non-decreasing until you
// invoke SherpaMnnCircularBufferReset() which resets head to 0.
SHERPA_ONNX_API int32_t
SherpaMnnCircularBufferHead(const SherpaMnnCircularBuffer *buffer);

// Clear all elements in the buffer
SHERPA_ONNX_API void SherpaMnnCircularBufferReset(
    const SherpaMnnCircularBuffer *buffer);

SHERPA_ONNX_API typedef struct SherpaMnnSpeechSegment {
  // The start index in samples of this segment
  int32_t start;

  // pointer to the array containing the samples
  float *samples;

  // number of samples in this segment
  int32_t n;
} SherpaMnnSpeechSegment;

typedef struct SherpaMnnVoiceActivityDetector SherpaMnnVoiceActivityDetector;

// Return an instance of VoiceActivityDetector.
// The user has to use SherpaMnnDestroyVoiceActivityDetector() to free
// the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnVoiceActivityDetector *
SherpaMnnCreateVoiceActivityDetector(const SherpaMnnVadModelConfig *config,
                                      float buffer_size_in_seconds);

SHERPA_ONNX_API void SherpaMnnDestroyVoiceActivityDetector(
    const SherpaMnnVoiceActivityDetector *p);

SHERPA_ONNX_API void SherpaMnnVoiceActivityDetectorAcceptWaveform(
    const SherpaMnnVoiceActivityDetector *p, const float *samples, int32_t n);

// Return 1 if there are no speech segments available.
// Return 0 if there are speech segments.
SHERPA_ONNX_API int32_t
SherpaMnnVoiceActivityDetectorEmpty(const SherpaMnnVoiceActivityDetector *p);

// Return 1 if there is voice detected.
// Return 0 if voice is silent.
SHERPA_ONNX_API int32_t SherpaMnnVoiceActivityDetectorDetected(
    const SherpaMnnVoiceActivityDetector *p);

// Return the first speech segment.
// It throws if SherpaMnnVoiceActivityDetectorEmpty() returns 1.
SHERPA_ONNX_API void SherpaMnnVoiceActivityDetectorPop(
    const SherpaMnnVoiceActivityDetector *p);

// Clear current speech segments.
SHERPA_ONNX_API void SherpaMnnVoiceActivityDetectorClear(
    const SherpaMnnVoiceActivityDetector *p);

// Return the first speech segment.
// The user has to use SherpaMnnDestroySpeechSegment() to free the returned
// pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnSpeechSegment *
SherpaMnnVoiceActivityDetectorFront(const SherpaMnnVoiceActivityDetector *p);

// Free the pointer returned SherpaMnnVoiceActivityDetectorFront().
SHERPA_ONNX_API void SherpaMnnDestroySpeechSegment(
    const SherpaMnnSpeechSegment *p);

// Re-initialize the voice activity detector.
SHERPA_ONNX_API void SherpaMnnVoiceActivityDetectorReset(
    const SherpaMnnVoiceActivityDetector *p);

SHERPA_ONNX_API void SherpaMnnVoiceActivityDetectorFlush(
    const SherpaMnnVoiceActivityDetector *p);

// ============================================================
// For offline Text-to-Speech (i.e., non-streaming TTS)
// ============================================================
SHERPA_ONNX_API typedef struct SherpaMnnOfflineTtsVitsModelConfig {
  const char *model;
  const char *lexicon;
  const char *tokens;
  const char *data_dir;

  float noise_scale;
  float noise_scale_w;
  float length_scale;  // < 1, faster in speech speed; > 1, slower in speed
  const char *dict_dir;
} SherpaMnnOfflineTtsVitsModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineTtsMatchaModelConfig {
  const char *acoustic_model;
  const char *vocoder;
  const char *lexicon;
  const char *tokens;
  const char *data_dir;

  float noise_scale;
  float length_scale;  // < 1, faster in speech speed; > 1, slower in speed
  const char *dict_dir;
} SherpaMnnOfflineTtsMatchaModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineTtsKokoroModelConfig {
  const char *model;
  const char *voices;
  const char *tokens;
  const char *data_dir;

  float length_scale;  // < 1, faster in speech speed; > 1, slower in speed
  const char *dict_dir;
  const char *lexicon;
} SherpaMnnOfflineTtsKokoroModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineTtsModelConfig {
  SherpaMnnOfflineTtsVitsModelConfig vits;
  int32_t num_threads;
  int32_t debug;
  const char *provider;
  SherpaMnnOfflineTtsMatchaModelConfig matcha;
  SherpaMnnOfflineTtsKokoroModelConfig kokoro;
} SherpaMnnOfflineTtsModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineTtsConfig {
  SherpaMnnOfflineTtsModelConfig model;
  const char *rule_fsts;
  int32_t max_num_sentences;
  const char *rule_fars;
  float silence_scale;
} SherpaMnnOfflineTtsConfig;

SHERPA_ONNX_API typedef struct SherpaMnnGeneratedAudio {
  const float *samples;  // in the range [-1, 1]
  int32_t n;             // number of samples
  int32_t sample_rate;
} SherpaMnnGeneratedAudio;

// If the callback returns 0, then it stops generating
// If the callback returns 1, then it keeps generating
typedef int32_t (*SherpaMnnGeneratedAudioCallback)(const float *samples,
                                                    int32_t n);

typedef int32_t (*SherpaMnnGeneratedAudioCallbackWithArg)(const float *samples,
                                                           int32_t n,
                                                           void *arg);

typedef int32_t (*SherpaMnnGeneratedAudioProgressCallback)(
    const float *samples, int32_t n, float p);

typedef int32_t (*SherpaMnnGeneratedAudioProgressCallbackWithArg)(
    const float *samples, int32_t n, float p, void *arg);

SHERPA_ONNX_API typedef struct SherpaMnnOfflineTts SherpaMnnOfflineTts;

// Create an instance of offline TTS. The user has to use DestroyOfflineTts()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOfflineTts *SherpaMnnCreateOfflineTts(
    const SherpaMnnOfflineTtsConfig *config);

// Free the pointer returned by SherpaMnnCreateOfflineTts()
SHERPA_ONNX_API void SherpaMnnDestroyOfflineTts(
    const SherpaMnnOfflineTts *tts);

// Return the sample rate of the current TTS object
SHERPA_ONNX_API int32_t
SherpaMnnOfflineTtsSampleRate(const SherpaMnnOfflineTts *tts);

// Return the number of speakers of the current TTS object
SHERPA_ONNX_API int32_t
SherpaMnnOfflineTtsNumSpeakers(const SherpaMnnOfflineTts *tts);

// Generate audio from the given text and speaker id (sid).
// The user has to use SherpaMnnDestroyOfflineTtsGeneratedAudio() to free the
// returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnGeneratedAudio *SherpaMnnOfflineTtsGenerate(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid,
    float speed);

// callback is called whenever SherpaMnnOfflineTtsConfig.max_num_sentences
// sentences have been processed. The pointer passed to the callback
// is freed once the callback is returned. So the caller should not keep
// a reference to it.
SHERPA_ONNX_API const SherpaMnnGeneratedAudio *
SherpaMnnOfflineTtsGenerateWithCallback(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioCallback callback);

SHERPA_ONNX_API
const SherpaMnnGeneratedAudio *
SherpaMnnOfflineTtsGenerateWithProgressCallback(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioProgressCallback callback);

SHERPA_ONNX_API
const SherpaMnnGeneratedAudio *
SherpaMnnOfflineTtsGenerateWithProgressCallbackWithArg(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioProgressCallbackWithArg callback, void *arg);

// Same as SherpaMnnGeneratedAudioCallback but you can pass an additional
// `void* arg` to the callback.
SHERPA_ONNX_API const SherpaMnnGeneratedAudio *
SherpaMnnOfflineTtsGenerateWithCallbackWithArg(
    const SherpaMnnOfflineTts *tts, const char *text, int32_t sid, float speed,
    SherpaMnnGeneratedAudioCallbackWithArg callback, void *arg);

SHERPA_ONNX_API void SherpaMnnDestroyOfflineTtsGeneratedAudio(
    const SherpaMnnGeneratedAudio *p);

// Write the generated audio to a wave file.
// The saved wave file contains a single channel and has 16-bit samples.
//
// Return 1 if the write succeeded; return 0 on failure.
SHERPA_ONNX_API int32_t SherpaMnnWriteWave(const float *samples, int32_t n,
                                            int32_t sample_rate,
                                            const char *filename);

// the amount of bytes needed to store a wave file which contains a
// single channel and has 16-bit samples.
SHERPA_ONNX_API int64_t SherpaMnnWaveFileSize(int32_t n_samples);

// Similar to SherpaMnnWriteWave , it writes wave to allocated  buffer;
//
// in some case (http tts api return wave binary file, server do not need to
// write wave to fs)
SHERPA_ONNX_API void SherpaMnnWriteWaveToBuffer(const float *samples,
                                                 int32_t n, int32_t sample_rate,
                                                 char *buffer);

SHERPA_ONNX_API typedef struct SherpaMnnWave {
  // samples normalized to the range [-1, 1]
  const float *samples;
  int32_t sample_rate;
  int32_t num_samples;
} SherpaMnnWave;

// Return a NULL pointer on error. It supports only standard WAVE file.
// Each sample should be 16-bit. It supports only single channel..
//
// If the returned pointer is not NULL, the user has to invoke
// SherpaMnnFreeWave() to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnWave *SherpaMnnReadWave(const char *filename);

// Similar to SherpaMnnReadWave(), it has read the content of `filename`
// into the array `data`.
//
// If the returned pointer is not NULL, the user has to invoke
// SherpaMnnFreeWave() to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnWave *SherpaMnnReadWaveFromBinaryData(
    const char *data, int32_t n);

SHERPA_ONNX_API void SherpaMnnFreeWave(const SherpaMnnWave *wave);

// ============================================================
// For spoken language identification
// ============================================================

SHERPA_ONNX_API typedef struct
    SherpaMnnSpokenLanguageIdentificationWhisperConfig {
  const char *encoder;
  const char *decoder;
  int32_t tail_paddings;
} SherpaMnnSpokenLanguageIdentificationWhisperConfig;

SHERPA_ONNX_API typedef struct SherpaMnnSpokenLanguageIdentificationConfig {
  SherpaMnnSpokenLanguageIdentificationWhisperConfig whisper;
  int32_t num_threads;
  int32_t debug;
  const char *provider;
} SherpaMnnSpokenLanguageIdentificationConfig;

SHERPA_ONNX_API typedef struct SherpaMnnSpokenLanguageIdentification
    SherpaMnnSpokenLanguageIdentification;

// Create an instance of SpokenLanguageIdentification.
// The user has to invoke SherpaMnnDestroySpokenLanguageIdentification()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnSpokenLanguageIdentification *
SherpaMnnCreateSpokenLanguageIdentification(
    const SherpaMnnSpokenLanguageIdentificationConfig *config);

SHERPA_ONNX_API void SherpaMnnDestroySpokenLanguageIdentification(
    const SherpaMnnSpokenLanguageIdentification *slid);

// The user has to invoke SherpaMnnDestroyOfflineStream()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API SherpaMnnOfflineStream *
SherpaMnnSpokenLanguageIdentificationCreateOfflineStream(
    const SherpaMnnSpokenLanguageIdentification *slid);

SHERPA_ONNX_API typedef struct SherpaMnnSpokenLanguageIdentificationResult {
  // en for English
  // de for German
  // zh for Chinese
  // es for Spanish
  // ...
  const char *lang;
} SherpaMnnSpokenLanguageIdentificationResult;

// The user has to invoke SherpaMnnDestroySpokenLanguageIdentificationResult()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnSpokenLanguageIdentificationResult *
SherpaMnnSpokenLanguageIdentificationCompute(
    const SherpaMnnSpokenLanguageIdentification *slid,
    const SherpaMnnOfflineStream *s);

SHERPA_ONNX_API void SherpaMnnDestroySpokenLanguageIdentificationResult(
    const SherpaMnnSpokenLanguageIdentificationResult *r);

// ============================================================
// For speaker embedding extraction
// ============================================================
SHERPA_ONNX_API typedef struct SherpaMnnSpeakerEmbeddingExtractorConfig {
  const char *model;
  int32_t num_threads;
  int32_t debug;
  const char *provider;
} SherpaMnnSpeakerEmbeddingExtractorConfig;

SHERPA_ONNX_API typedef struct SherpaMnnSpeakerEmbeddingExtractor
    SherpaMnnSpeakerEmbeddingExtractor;

// The user has to invoke SherpaMnnDestroySpeakerEmbeddingExtractor()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnSpeakerEmbeddingExtractor *
SherpaMnnCreateSpeakerEmbeddingExtractor(
    const SherpaMnnSpeakerEmbeddingExtractorConfig *config);

SHERPA_ONNX_API void SherpaMnnDestroySpeakerEmbeddingExtractor(
    const SherpaMnnSpeakerEmbeddingExtractor *p);

SHERPA_ONNX_API int32_t SherpaMnnSpeakerEmbeddingExtractorDim(
    const SherpaMnnSpeakerEmbeddingExtractor *p);

// The user has to invoke SherpaMnnDestroyOnlineStream() to free the returned
// pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnOnlineStream *
SherpaMnnSpeakerEmbeddingExtractorCreateStream(
    const SherpaMnnSpeakerEmbeddingExtractor *p);

// Return 1 if the stream has enough feature frames for computing embeddings.
// Return 0 otherwise.
SHERPA_ONNX_API int32_t SherpaMnnSpeakerEmbeddingExtractorIsReady(
    const SherpaMnnSpeakerEmbeddingExtractor *p,
    const SherpaMnnOnlineStream *s);

// Compute the embedding of the stream.
//
// @return Return a pointer pointing to an array containing the embedding.
// The length of the array is `dim` as returned by
// SherpaMnnSpeakerEmbeddingExtractorDim(p)
//
// The user has to invoke SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const float *
SherpaMnnSpeakerEmbeddingExtractorComputeEmbedding(
    const SherpaMnnSpeakerEmbeddingExtractor *p,
    const SherpaMnnOnlineStream *s);

SHERPA_ONNX_API void SherpaMnnSpeakerEmbeddingExtractorDestroyEmbedding(
    const float *v);

SHERPA_ONNX_API typedef struct SherpaMnnSpeakerEmbeddingManager
    SherpaMnnSpeakerEmbeddingManager;

// The user has to invoke SherpaMnnDestroySpeakerEmbeddingManager()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnSpeakerEmbeddingManager *
SherpaMnnCreateSpeakerEmbeddingManager(int32_t dim);

SHERPA_ONNX_API void SherpaMnnDestroySpeakerEmbeddingManager(
    const SherpaMnnSpeakerEmbeddingManager *p);

// Register the embedding of a user
//
// @param name  The name of the user
// @param p Pointer to an array containing the embeddings. The length of the
//          array must be equal to `dim` used to construct the manager `p`.
//
// @return Return 1 if added successfully. Return 0 on error
SHERPA_ONNX_API int32_t
SherpaMnnSpeakerEmbeddingManagerAdd(const SherpaMnnSpeakerEmbeddingManager *p,
                                     const char *name, const float *v);

// @param v Pointer to an array of embeddings. If there are n embeddings, then
//          v[0] is the pointer to the 0-th array containing the embeddings
//          v[1] is the pointer to the 1-st array containing the embeddings
//          v[n-1] is the pointer to the last array containing the embeddings
//          v[n] is a NULL pointer
// @return Return 1 if added successfully. Return 0 on error
SHERPA_ONNX_API int32_t SherpaMnnSpeakerEmbeddingManagerAddList(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name,
    const float **v);

// Similar to SherpaMnnSpeakerEmbeddingManagerAddList() but the memory
// is flattened.
//
// The length of the input array should be `n * dim`.
//
// @return Return 1 if added successfully. Return 0 on error
SHERPA_ONNX_API int32_t SherpaMnnSpeakerEmbeddingManagerAddListFlattened(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name,
    const float *v, int32_t n);

// Remove a user.
// @param naem The name of the user to remove.
// @return Return 1 if removed successfully; return 0 on error.
//
// Note if the user does not exist, it also returns 0.
SHERPA_ONNX_API int32_t SherpaMnnSpeakerEmbeddingManagerRemove(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name);

// Search if an existing users' embedding matches the given one.
//
// @param p Pointer to an array containing the embedding. The dim
//          of the array must equal to `dim` used to construct the manager `p`.
// @param threshold A value between 0 and 1. If the similarity score exceeds
//                  this threshold, we say a match is found.
// @return Returns the name of the user if found. Return NULL if not found.
//         If not NULL, the caller has to invoke
//          SherpaMnnSpeakerEmbeddingManagerFreeSearch() to free the returned
//          pointer to avoid memory leak.
SHERPA_ONNX_API const char *SherpaMnnSpeakerEmbeddingManagerSearch(
    const SherpaMnnSpeakerEmbeddingManager *p, const float *v,
    float threshold);

SHERPA_ONNX_API void SherpaMnnSpeakerEmbeddingManagerFreeSearch(
    const char *name);

SHERPA_ONNX_API typedef struct SherpaMnnSpeakerEmbeddingManagerSpeakerMatch {
  float score;
  const char *name;
} SherpaMnnSpeakerEmbeddingManagerSpeakerMatch;

SHERPA_ONNX_API typedef struct
    SherpaMnnSpeakerEmbeddingManagerBestMatchesResult {
  const SherpaMnnSpeakerEmbeddingManagerSpeakerMatch *matches;
  int32_t count;
} SherpaMnnSpeakerEmbeddingManagerBestMatchesResult;

// Get the best matching speakers whose embeddings match the given
// embedding.
//
// @param p Pointer to the SherpaMnnSpeakerEmbeddingManager instance.
// @param v Pointer to an array containing the embedding vector.
// @param threshold Minimum similarity score required for a match (between 0 and
// 1).
// @param n Number of best matches to retrieve.
// @return Returns a pointer to
// SherpaMnnSpeakerEmbeddingManagerBestMatchesResult
//         containing the best matches found. Returns NULL if no matches are
//         found. The caller is responsible for freeing the returned pointer
//         using SherpaMnnSpeakerEmbeddingManagerFreeBestMatches() to
//         avoid memory leaks.
SHERPA_ONNX_API const SherpaMnnSpeakerEmbeddingManagerBestMatchesResult *
SherpaMnnSpeakerEmbeddingManagerGetBestMatches(
    const SherpaMnnSpeakerEmbeddingManager *p, const float *v, float threshold,
    int32_t n);

SHERPA_ONNX_API void SherpaMnnSpeakerEmbeddingManagerFreeBestMatches(
    const SherpaMnnSpeakerEmbeddingManagerBestMatchesResult *r);

// Check whether the input embedding matches the embedding of the input
// speaker.
//
// It is for speaker verification.
//
// @param name The target speaker name.
// @param p The input embedding to check.
// @param threshold A value between 0 and 1.
// @return Return 1 if it matches. Otherwise, it returns 0.
SHERPA_ONNX_API int32_t SherpaMnnSpeakerEmbeddingManagerVerify(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name,
    const float *v, float threshold);

// Return 1 if the user with the name is in the manager.
// Return 0 if the user does not exist.
SHERPA_ONNX_API int32_t SherpaMnnSpeakerEmbeddingManagerContains(
    const SherpaMnnSpeakerEmbeddingManager *p, const char *name);

// Return number of speakers in the manager.
SHERPA_ONNX_API int32_t SherpaMnnSpeakerEmbeddingManagerNumSpeakers(
    const SherpaMnnSpeakerEmbeddingManager *p);

// Return the name of all speakers in the manager.
//
// @return Return an array of pointers `ans`. If there are n speakers, then
// - ans[0] contains the name of the 0-th speaker
// - ans[1] contains the name of the 1-st speaker
// - ans[n-1] contains the name of the last speaker
// - ans[n] is NULL
// If there are no users at all, then ans[0] is NULL. In any case,
// `ans` is not NULL.
//
// Each name is NULL-terminated
//
// The caller has to invoke SherpaMnnSpeakerEmbeddingManagerFreeAllSpeakers()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const char *const *
SherpaMnnSpeakerEmbeddingManagerGetAllSpeakers(
    const SherpaMnnSpeakerEmbeddingManager *p);

SHERPA_ONNX_API void SherpaMnnSpeakerEmbeddingManagerFreeAllSpeakers(
    const char *const *names);

// ============================================================
// For audio tagging
// ============================================================
SHERPA_ONNX_API typedef struct
    SherpaMnnOfflineZipformerAudioTaggingModelConfig {
  const char *model;
} SherpaMnnOfflineZipformerAudioTaggingModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnAudioTaggingModelConfig {
  SherpaMnnOfflineZipformerAudioTaggingModelConfig zipformer;
  const char *ced;
  int32_t num_threads;
  int32_t debug;  // true to print debug information of the model
  const char *provider;
} SherpaMnnAudioTaggingModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnAudioTaggingConfig {
  SherpaMnnAudioTaggingModelConfig model;
  const char *labels;
  int32_t top_k;
} SherpaMnnAudioTaggingConfig;

SHERPA_ONNX_API typedef struct SherpaMnnAudioEvent {
  const char *name;
  int32_t index;
  float prob;
} SherpaMnnAudioEvent;

SHERPA_ONNX_API typedef struct SherpaMnnAudioTagging SherpaMnnAudioTagging;

// The user has to invoke
// SherpaMnnDestroyAudioTagging()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnAudioTagging *SherpaMnnCreateAudioTagging(
    const SherpaMnnAudioTaggingConfig *config);

SHERPA_ONNX_API void SherpaMnnDestroyAudioTagging(
    const SherpaMnnAudioTagging *tagger);

// The user has to invoke SherpaMnnDestroyOfflineStream()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnOfflineStream *
SherpaMnnAudioTaggingCreateOfflineStream(const SherpaMnnAudioTagging *tagger);

// Return an array of pointers. The length of the array is top_k + 1.
// If top_k is -1, then config.top_k is used, where config is the config
// used to create the input tagger.
//
// The ans[0]->prob has the largest probability among the array elements
// The last element of the array is a null pointer
//
// The user has to use SherpaMnnAudioTaggingFreeResults()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnAudioEvent *const *
SherpaMnnAudioTaggingCompute(const SherpaMnnAudioTagging *tagger,
                              const SherpaMnnOfflineStream *s, int32_t top_k);

SHERPA_ONNX_API void SherpaMnnAudioTaggingFreeResults(
    const SherpaMnnAudioEvent *const *p);

// ============================================================
// For punctuation
// ============================================================

SHERPA_ONNX_API typedef struct SherpaMnnOfflinePunctuationModelConfig {
  const char *ct_transformer;
  int32_t num_threads;
  int32_t debug;  // true to print debug information of the model
  const char *provider;
} SherpaMnnOfflinePunctuationModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflinePunctuationConfig {
  SherpaMnnOfflinePunctuationModelConfig model;
} SherpaMnnOfflinePunctuationConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflinePunctuation
    SherpaMnnOfflinePunctuation;

// The user has to invoke SherpaMnnDestroyOfflinePunctuation()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnOfflinePunctuation *
SherpaMnnCreateOfflinePunctuation(
    const SherpaMnnOfflinePunctuationConfig *config);

SHERPA_ONNX_API void SherpaMnnDestroyOfflinePunctuation(
    const SherpaMnnOfflinePunctuation *punct);

// Add punctuations to the input text.
// The user has to invoke SherpaOfflinePunctuationFreeText()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const char *SherpaOfflinePunctuationAddPunct(
    const SherpaMnnOfflinePunctuation *punct, const char *text);

SHERPA_ONNX_API void SherpaOfflinePunctuationFreeText(const char *text);

SHERPA_ONNX_API typedef struct SherpaMnnOnlinePunctuationModelConfig {
  const char *cnn_bilstm;
  const char *bpe_vocab;
  int32_t num_threads;
  int32_t debug;
  const char *provider;
} SherpaMnnOnlinePunctuationModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOnlinePunctuationConfig {
  SherpaMnnOnlinePunctuationModelConfig model;
} SherpaMnnOnlinePunctuationConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOnlinePunctuation
    SherpaMnnOnlinePunctuation;

// Create an online punctuation processor. The user has to invoke
// SherpaMnnDestroyOnlinePunctuation() to free the returned pointer
// to avoid memory leak
SHERPA_ONNX_API const SherpaMnnOnlinePunctuation *
SherpaMnnCreateOnlinePunctuation(
    const SherpaMnnOnlinePunctuationConfig *config);

// Free a pointer returned by SherpaMnnCreateOnlinePunctuation()
SHERPA_ONNX_API void SherpaMnnDestroyOnlinePunctuation(
    const SherpaMnnOnlinePunctuation *punctuation);

// Add punctuations to the input text. The user has to invoke
// SherpaMnnOnlinePunctuationFreeText() to free the returned pointer
// to avoid memory leak
SHERPA_ONNX_API const char *SherpaMnnOnlinePunctuationAddPunct(
    const SherpaMnnOnlinePunctuation *punctuation, const char *text);

// Free a pointer returned by SherpaMnnOnlinePunctuationAddPunct()
SHERPA_ONNX_API void SherpaMnnOnlinePunctuationFreeText(const char *text);

// for resampling
SHERPA_ONNX_API typedef struct SherpaMnnLinearResampler
    SherpaMnnLinearResampler;

/*
      float min_freq = min(sampling_rate_in_hz, samp_rate_out_hz);
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;
      int32_t lowpass_filter_width = 6;

      You can set filter_cutoff_hz to lowpass_cutoff
      sand set num_zeros to lowpass_filter_width
*/
// The user has to invoke SherpaMnnDestroyLinearResampler()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnLinearResampler *
SherpaMnnCreateLinearResampler(int32_t samp_rate_in_hz,
                                int32_t samp_rate_out_hz,
                                float filter_cutoff_hz, int32_t num_zeros);

SHERPA_ONNX_API void SherpaMnnDestroyLinearResampler(
    const SherpaMnnLinearResampler *p);

SHERPA_ONNX_API void SherpaMnnLinearResamplerReset(
    const SherpaMnnLinearResampler *p);

typedef struct SherpaMnnResampleOut {
  const float *samples;
  int32_t n;
} SherpaMnnResampleOut;
// The user has to invoke SherpaMnnLinearResamplerResampleFree()
// to free the returned pointer to avoid memory leak.
//
// If this is the last segment, you can set flush to 1; otherwise, please
// set flush to 0
SHERPA_ONNX_API const SherpaMnnResampleOut *SherpaMnnLinearResamplerResample(
    const SherpaMnnLinearResampler *p, const float *input, int32_t input_dim,
    int32_t flush);

SHERPA_ONNX_API void SherpaMnnLinearResamplerResampleFree(
    const SherpaMnnResampleOut *p);

SHERPA_ONNX_API int32_t SherpaMnnLinearResamplerResampleGetInputSampleRate(
    const SherpaMnnLinearResampler *p);

SHERPA_ONNX_API int32_t SherpaMnnLinearResamplerResampleGetOutputSampleRate(
    const SherpaMnnLinearResampler *p);

// Return 1 if the file exists; return 0 if the file does not exist.
SHERPA_ONNX_API int32_t SherpaMnnFileExists(const char *filename);

// =========================================================================
// For offline speaker diarization (i.e., non-streaming speaker diarization)
// =========================================================================
SHERPA_ONNX_API typedef struct
    SherpaMnnOfflineSpeakerSegmentationPyannoteModelConfig {
  const char *model;
} SherpaMnnOfflineSpeakerSegmentationPyannoteModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineSpeakerSegmentationModelConfig {
  SherpaMnnOfflineSpeakerSegmentationPyannoteModelConfig pyannote;
  int32_t num_threads;   // 1
  int32_t debug;         // false
  const char *provider;  // "cpu"
} SherpaMnnOfflineSpeakerSegmentationModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnFastClusteringConfig {
  // If greater than 0, then threshold is ignored.
  //
  // We strongly recommend that you set it if you know the number of clusters
  // in advance
  int32_t num_clusters;

  // distance threshold.
  //
  // The smaller, the more clusters it will generate.
  // The larger, the fewer clusters it will generate.
  float threshold;
} SherpaMnnFastClusteringConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineSpeakerDiarizationConfig {
  SherpaMnnOfflineSpeakerSegmentationModelConfig segmentation;
  SherpaMnnSpeakerEmbeddingExtractorConfig embedding;
  SherpaMnnFastClusteringConfig clustering;

  // if a segment is less than this value, then it is discarded
  float min_duration_on;  // in seconds

  // if the gap between to segments of the same speaker is less than this value,
  // then these two segments are merged into a single segment.
  // We do this recursively.
  float min_duration_off;  // in seconds
} SherpaMnnOfflineSpeakerDiarizationConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineSpeakerDiarization
    SherpaMnnOfflineSpeakerDiarization;

// The users has to invoke SherpaMnnDestroyOfflineSpeakerDiarization()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnOfflineSpeakerDiarization *
SherpaMnnCreateOfflineSpeakerDiarization(
    const SherpaMnnOfflineSpeakerDiarizationConfig *config);

// Free the pointer returned by SherpaMnnCreateOfflineSpeakerDiarization()
SHERPA_ONNX_API void SherpaMnnDestroyOfflineSpeakerDiarization(
    const SherpaMnnOfflineSpeakerDiarization *sd);

// Expected sample rate of the input audio samples
SHERPA_ONNX_API int32_t SherpaMnnOfflineSpeakerDiarizationGetSampleRate(
    const SherpaMnnOfflineSpeakerDiarization *sd);

// Only config->clustering is used. All other fields are ignored
SHERPA_ONNX_API void SherpaMnnOfflineSpeakerDiarizationSetConfig(
    const SherpaMnnOfflineSpeakerDiarization *sd,
    const SherpaMnnOfflineSpeakerDiarizationConfig *config);

SHERPA_ONNX_API typedef struct SherpaMnnOfflineSpeakerDiarizationResult
    SherpaMnnOfflineSpeakerDiarizationResult;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineSpeakerDiarizationSegment {
  float start;
  float end;
  int32_t speaker;
} SherpaMnnOfflineSpeakerDiarizationSegment;

SHERPA_ONNX_API int32_t SherpaMnnOfflineSpeakerDiarizationResultGetNumSpeakers(
    const SherpaMnnOfflineSpeakerDiarizationResult *r);

SHERPA_ONNX_API int32_t SherpaMnnOfflineSpeakerDiarizationResultGetNumSegments(
    const SherpaMnnOfflineSpeakerDiarizationResult *r);

// The user has to invoke SherpaMnnOfflineSpeakerDiarizationDestroySegment()
// to free the returned pointer to avoid memory leak.
//
// The returned pointer is the start address of an array.
// Number of entries in the array equals to the value
// returned by SherpaMnnOfflineSpeakerDiarizationResultGetNumSegments()
SHERPA_ONNX_API const SherpaMnnOfflineSpeakerDiarizationSegment *
SherpaMnnOfflineSpeakerDiarizationResultSortByStartTime(
    const SherpaMnnOfflineSpeakerDiarizationResult *r);

SHERPA_ONNX_API void SherpaMnnOfflineSpeakerDiarizationDestroySegment(
    const SherpaMnnOfflineSpeakerDiarizationSegment *s);

typedef int32_t (*SherpaMnnOfflineSpeakerDiarizationProgressCallback)(
    int32_t num_processed_chunks, int32_t num_total_chunks, void *arg);

typedef int32_t (*SherpaMnnOfflineSpeakerDiarizationProgressCallbackNoArg)(
    int32_t num_processed_chunks, int32_t num_total_chunks);

// The user has to invoke SherpaMnnOfflineSpeakerDiarizationDestroyResult()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOfflineSpeakerDiarizationResult *
SherpaMnnOfflineSpeakerDiarizationProcess(
    const SherpaMnnOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n);

// The user has to invoke SherpaMnnOfflineSpeakerDiarizationDestroyResult()
// to free the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOfflineSpeakerDiarizationResult *
SherpaMnnOfflineSpeakerDiarizationProcessWithCallback(
    const SherpaMnnOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n, SherpaMnnOfflineSpeakerDiarizationProgressCallback callback,
    void *arg);

SHERPA_ONNX_API const SherpaMnnOfflineSpeakerDiarizationResult *
SherpaMnnOfflineSpeakerDiarizationProcessWithCallbackNoArg(
    const SherpaMnnOfflineSpeakerDiarization *sd, const float *samples,
    int32_t n,
    SherpaMnnOfflineSpeakerDiarizationProgressCallbackNoArg callback);

SHERPA_ONNX_API void SherpaMnnOfflineSpeakerDiarizationDestroyResult(
    const SherpaMnnOfflineSpeakerDiarizationResult *r);

// =========================================================================
// For offline speech enhancement
// =========================================================================
SHERPA_ONNX_API typedef struct SherpaMnnOfflineSpeechDenoiserGtcrnModelConfig {
  const char *model;
} SherpaMnnOfflineSpeechDenoiserGtcrnModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineSpeechDenoiserModelConfig {
  SherpaMnnOfflineSpeechDenoiserGtcrnModelConfig gtcrn;
  int32_t num_threads;
  int32_t debug;  // true to print debug information of the model
  const char *provider;
} SherpaMnnOfflineSpeechDenoiserModelConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineSpeechDenoiserConfig {
  SherpaMnnOfflineSpeechDenoiserModelConfig model;
} SherpaMnnOfflineSpeechDenoiserConfig;

SHERPA_ONNX_API typedef struct SherpaMnnOfflineSpeechDenoiser
    SherpaMnnOfflineSpeechDenoiser;

// The users has to invoke SherpaMnnDestroyOfflineSpeechDenoiser()
// to free the returned pointer to avoid memory leak
SHERPA_ONNX_API const SherpaMnnOfflineSpeechDenoiser *
SherpaMnnCreateOfflineSpeechDenoiser(
    const SherpaMnnOfflineSpeechDenoiserConfig *config);

// Free the pointer returned by SherpaMnnCreateOfflineSpeechDenoiser()
SHERPA_ONNX_API void SherpaMnnDestroyOfflineSpeechDenoiser(
    const SherpaMnnOfflineSpeechDenoiser *sd);

SHERPA_ONNX_API int32_t SherpaMnnOfflineSpeechDenoiserGetSampleRate(
    const SherpaMnnOfflineSpeechDenoiser *sd);

SHERPA_ONNX_API typedef struct SherpaMnnDenoisedAudio {
  const float *samples;  // in the range [-1, 1]
  int32_t n;             // number of samples
  int32_t sample_rate;
} SherpaMnnDenoisedAudio;

// Run speech denosing on input samples
// @param samples  A 1-D array containing the input audio samples. Each sample
//           should be in the range [-1, 1].
// @param n  Number of samples
// @param sample_rate Sample rate of the input samples
//
// The user MUST use SherpaMnnDestroyDenoisedAudio() to free the returned
// pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnDenoisedAudio *
SherpaMnnOfflineSpeechDenoiserRun(const SherpaMnnOfflineSpeechDenoiser *sd,
                                   const float *samples, int32_t n,
                                   int32_t sample_rate);

SHERPA_ONNX_API void SherpaMnnDestroyDenoisedAudio(
    const SherpaMnnDenoisedAudio *p);

#ifdef __OHOS__

// It is for HarmonyOS
typedef struct NativeResourceManager NativeResourceManager;

SHERPA_ONNX_API const SherpaMnnOfflineSpeechDenoiser *
SherpaMnnCreateOfflineSpeechDenoiserOHOS(
    const SherpaMnnOfflineSpeechDenoiserConfig *config,
    NativeResourceManager *mgr);

/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          SherpaMnnDestroyOnlineRecognizer() to free it to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnOnlineRecognizer *
SherpaMnnCreateOnlineRecognizerOHOS(
    const SherpaMnnOnlineRecognizerConfig *config, NativeResourceManager *mgr);

/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          SherpaMnnDestroyOfflineRecognizer() to free it to avoid memory
//          leak.
SHERPA_ONNX_API const SherpaMnnOfflineRecognizer *
SherpaMnnCreateOfflineRecognizerOHOS(
    const SherpaMnnOfflineRecognizerConfig *config,
    NativeResourceManager *mgr);

// Return an instance of VoiceActivityDetector.
// The user has to use SherpaMnnDestroyVoiceActivityDetector() to free
// the returned pointer to avoid memory leak.
SHERPA_ONNX_API const SherpaMnnVoiceActivityDetector *
SherpaMnnCreateVoiceActivityDetectorOHOS(
    const SherpaMnnVadModelConfig *config, float buffer_size_in_seconds,
    NativeResourceManager *mgr);

SHERPA_ONNX_API const SherpaMnnOfflineTts *SherpaMnnCreateOfflineTtsOHOS(
    const SherpaMnnOfflineTtsConfig *config, NativeResourceManager *mgr);

SHERPA_ONNX_API const SherpaMnnSpeakerEmbeddingExtractor *
SherpaMnnCreateSpeakerEmbeddingExtractorOHOS(
    const SherpaMnnSpeakerEmbeddingExtractorConfig *config,
    NativeResourceManager *mgr);

SHERPA_ONNX_API const SherpaMnnKeywordSpotter *
SherpaMnnCreateKeywordSpotterOHOS(const SherpaMnnKeywordSpotterConfig *config,
                                   NativeResourceManager *mgr);

SHERPA_ONNX_API const SherpaMnnOfflineSpeakerDiarization *
SherpaMnnCreateOfflineSpeakerDiarizationOHOS(
    const SherpaMnnOfflineSpeakerDiarizationConfig *config,
    NativeResourceManager *mgr);
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // SHERPA_ONNX_C_API_C_API_H_
