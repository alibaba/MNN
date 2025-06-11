include_directories(${CMAKE_SOURCE_DIR})

add_executable(streaming-zipformer-cxx-api ./streaming-zipformer-cxx-api.cc)
target_link_libraries(streaming-zipformer-cxx-api sherpa-mnn-cxx-api)

add_executable(speech-enhancement-gtcrn-cxx-api ./speech-enhancement-gtcrn-cxx-api.cc)
target_link_libraries(speech-enhancement-gtcrn-cxx-api sherpa-mnn-cxx-api)

add_executable(kws-cxx-api ./kws-cxx-api.cc)
target_link_libraries(kws-cxx-api sherpa-mnn-cxx-api)

add_executable(streaming-zipformer-rtf-cxx-api ./streaming-zipformer-rtf-cxx-api.cc)
target_link_libraries(streaming-zipformer-rtf-cxx-api sherpa-mnn-cxx-api)

add_executable(whisper-cxx-api ./whisper-cxx-api.cc)
target_link_libraries(whisper-cxx-api sherpa-mnn-cxx-api)

add_executable(fire-red-asr-cxx-api ./fire-red-asr-cxx-api.cc)
target_link_libraries(fire-red-asr-cxx-api sherpa-mnn-cxx-api)

add_executable(moonshine-cxx-api ./moonshine-cxx-api.cc)
target_link_libraries(moonshine-cxx-api sherpa-mnn-cxx-api)

add_executable(sense-voice-cxx-api ./sense-voice-cxx-api.cc)
target_link_libraries(sense-voice-cxx-api sherpa-mnn-cxx-api)

if(SHERPA_MNN_ENABLE_TTS)
  add_executable(matcha-tts-zh-cxx-api ./matcha-tts-zh-cxx-api.cc)
  target_link_libraries(matcha-tts-zh-cxx-api sherpa-mnn-cxx-api)

  add_executable(matcha-tts-en-cxx-api ./matcha-tts-en-cxx-api.cc)
  target_link_libraries(matcha-tts-en-cxx-api sherpa-mnn-cxx-api)

  add_executable(kokoro-tts-en-cxx-api ./kokoro-tts-en-cxx-api.cc)
  target_link_libraries(kokoro-tts-en-cxx-api sherpa-mnn-cxx-api)

  add_executable(kokoro-tts-zh-en-cxx-api ./kokoro-tts-zh-en-cxx-api.cc)
  target_link_libraries(kokoro-tts-zh-en-cxx-api sherpa-mnn-cxx-api)
endif()
