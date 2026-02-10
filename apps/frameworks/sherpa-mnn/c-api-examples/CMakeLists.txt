include(cargs)

include_directories(${CMAKE_SOURCE_DIR})
add_executable(decode-file-c-api decode-file-c-api.c)
target_link_libraries(decode-file-c-api sherpa-mnn-c-api cargs)

add_executable(kws-c-api kws-c-api.c)
target_link_libraries(kws-c-api sherpa-mnn-c-api)

add_executable(speech-enhancement-gtcrn-c-api speech-enhancement-gtcrn-c-api.c)
target_link_libraries(speech-enhancement-gtcrn-c-api sherpa-mnn-c-api)

if(SHERPA_MNN_ENABLE_TTS)
  add_executable(offline-tts-c-api offline-tts-c-api.c)
  target_link_libraries(offline-tts-c-api sherpa-mnn-c-api cargs)

  add_executable(matcha-tts-zh-c-api matcha-tts-zh-c-api.c)
  target_link_libraries(matcha-tts-zh-c-api sherpa-mnn-c-api)

  add_executable(matcha-tts-en-c-api matcha-tts-en-c-api.c)
  target_link_libraries(matcha-tts-en-c-api sherpa-mnn-c-api)

  add_executable(kokoro-tts-en-c-api kokoro-tts-en-c-api.c)
  target_link_libraries(kokoro-tts-en-c-api sherpa-mnn-c-api)

  add_executable(kokoro-tts-zh-en-c-api kokoro-tts-zh-en-c-api.c)
  target_link_libraries(kokoro-tts-zh-en-c-api sherpa-mnn-c-api)
endif()

if(SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION)
  add_executable(offline-speaker-diarization-c-api offline-speaker-diarization-c-api.c)
  target_link_libraries(offline-speaker-diarization-c-api sherpa-mnn-c-api)
endif()

add_executable(spoken-language-identification-c-api spoken-language-identification-c-api.c)
target_link_libraries(spoken-language-identification-c-api sherpa-mnn-c-api)

add_executable(speaker-identification-c-api speaker-identification-c-api.c)
target_link_libraries(speaker-identification-c-api sherpa-mnn-c-api)

add_executable(streaming-hlg-decode-file-c-api streaming-hlg-decode-file-c-api.c)
target_link_libraries(streaming-hlg-decode-file-c-api sherpa-mnn-c-api)

add_executable(audio-tagging-c-api audio-tagging-c-api.c)
target_link_libraries(audio-tagging-c-api sherpa-mnn-c-api)

add_executable(add-punctuation-c-api add-punctuation-c-api.c)
target_link_libraries(add-punctuation-c-api sherpa-mnn-c-api)

add_executable(whisper-c-api whisper-c-api.c)
target_link_libraries(whisper-c-api sherpa-mnn-c-api)

add_executable(fire-red-asr-c-api fire-red-asr-c-api.c)
target_link_libraries(fire-red-asr-c-api sherpa-mnn-c-api)

add_executable(sense-voice-c-api sense-voice-c-api.c)
target_link_libraries(sense-voice-c-api sherpa-mnn-c-api)

add_executable(moonshine-c-api moonshine-c-api.c)
target_link_libraries(moonshine-c-api sherpa-mnn-c-api)

add_executable(zipformer-c-api zipformer-c-api.c)
target_link_libraries(zipformer-c-api sherpa-mnn-c-api)

add_executable(streaming-zipformer-c-api streaming-zipformer-c-api.c)
target_link_libraries(streaming-zipformer-c-api sherpa-mnn-c-api)

add_executable(paraformer-c-api paraformer-c-api.c)
target_link_libraries(paraformer-c-api sherpa-mnn-c-api)

add_executable(streaming-paraformer-c-api streaming-paraformer-c-api.c)
target_link_libraries(streaming-paraformer-c-api sherpa-mnn-c-api)

add_executable(telespeech-c-api telespeech-c-api.c)
target_link_libraries(telespeech-c-api sherpa-mnn-c-api)

add_executable(vad-sense-voice-c-api vad-sense-voice-c-api.c)
target_link_libraries(vad-sense-voice-c-api sherpa-mnn-c-api)

add_executable(vad-whisper-c-api vad-whisper-c-api.c)
target_link_libraries(vad-whisper-c-api sherpa-mnn-c-api)

add_executable(vad-moonshine-c-api vad-moonshine-c-api.c)
target_link_libraries(vad-moonshine-c-api sherpa-mnn-c-api)

add_executable(streaming-zipformer-buffered-tokens-hotwords-c-api
               streaming-zipformer-buffered-tokens-hotwords-c-api.c)
target_link_libraries(streaming-zipformer-buffered-tokens-hotwords-c-api sherpa-mnn-c-api)

add_executable(streaming-paraformer-buffered-tokens-c-api
               streaming-paraformer-buffered-tokens-c-api.c)
target_link_libraries(streaming-paraformer-buffered-tokens-c-api sherpa-mnn-c-api)

add_executable(streaming-ctc-buffered-tokens-c-api
               streaming-ctc-buffered-tokens-c-api.c)
target_link_libraries(streaming-ctc-buffered-tokens-c-api sherpa-mnn-c-api)

add_executable(keywords-spotter-buffered-tokens-keywords-c-api
               keywords-spotter-buffered-tokens-keywords-c-api.c)
target_link_libraries(keywords-spotter-buffered-tokens-keywords-c-api sherpa-mnn-c-api)

if(SHERPA_MNN_HAS_ALSA)
  add_subdirectory(./asr-microphone-example)
elseif((UNIX AND NOT APPLE) OR LINUX)
  message(WARNING "Not include ./asr-microphone-example since alsa is not available")
endif()
