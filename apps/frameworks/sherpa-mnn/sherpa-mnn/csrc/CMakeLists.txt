include_directories(${CMAKE_SOURCE_DIR})
message(STATUS "Currnet: ${CMAKE_SOURCE_DIR}")

if(SHERPA_MNN_ENABLE_PYTHON)
  message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import sys; print('.'.join(sys.version.split('.')[:2]))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_VERSION
  )
  message(STATUS "PYTHON_VERSION: ${PYTHON_VERSION}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(sources
  base64-decode.cc
  bbpe.cc
  cat.cc
  circular-buffer.cc
  context-graph.cc
  endpoint.cc
  features.cc
  file-utils.cc
  fst-utils.cc
  hypothesis.cc
  keyword-spotter-impl.cc
  keyword-spotter.cc
  offline-ctc-fst-decoder-config.cc
  offline-ctc-fst-decoder.cc
  offline-ctc-greedy-search-decoder.cc
  offline-ctc-model.cc
  offline-fire-red-asr-greedy-search-decoder.cc
  offline-fire-red-asr-model-config.cc
  offline-fire-red-asr-model.cc
  offline-lm-config.cc
  offline-lm.cc
  offline-model-config.cc
  offline-moonshine-greedy-search-decoder.cc
  offline-moonshine-model-config.cc
  offline-moonshine-model.cc
  offline-nemo-enc-dec-ctc-model-config.cc
  offline-nemo-enc-dec-ctc-model.cc
  offline-paraformer-greedy-search-decoder.cc
  offline-paraformer-model-config.cc
  offline-paraformer-model.cc
  offline-recognizer-impl.cc
  offline-recognizer.cc
  offline-rnn-lm.cc
  offline-sense-voice-model-config.cc
  offline-sense-voice-model.cc
  offline-stream.cc
  offline-tdnn-ctc-model.cc
  offline-tdnn-model-config.cc
  offline-telespeech-ctc-model.cc
  offline-transducer-greedy-search-decoder.cc
  offline-transducer-greedy-search-nemo-decoder.cc
  offline-transducer-model-config.cc
  offline-transducer-model.cc
  offline-transducer-modified-beam-search-decoder.cc
  offline-transducer-nemo-model.cc
  offline-wenet-ctc-model-config.cc
  offline-wenet-ctc-model.cc
  offline-whisper-greedy-search-decoder.cc
  offline-whisper-model-config.cc
  offline-whisper-model.cc
  offline-zipformer-ctc-model-config.cc
  offline-zipformer-ctc-model.cc
  online-conformer-transducer-model.cc
  online-ctc-fst-decoder-config.cc
  online-ctc-fst-decoder.cc
  online-ctc-greedy-search-decoder.cc
  online-ctc-model.cc
  online-ebranchformer-transducer-model.cc
  online-lm-config.cc
  online-lm.cc
  online-lstm-transducer-model.cc
  online-model-config.cc
  online-nemo-ctc-model-config.cc
  online-nemo-ctc-model.cc
  online-paraformer-model-config.cc
  online-paraformer-model.cc
  online-recognizer-impl.cc
  online-recognizer.cc
  online-rnn-lm.cc
  online-stream.cc
  online-transducer-decoder.cc
  online-transducer-greedy-search-decoder.cc
  online-transducer-greedy-search-nemo-decoder.cc
  online-transducer-model-config.cc
  online-transducer-model.cc
  online-transducer-modified-beam-search-decoder.cc
  online-transducer-nemo-model.cc
  online-wenet-ctc-model-config.cc
  online-wenet-ctc-model.cc
  online-zipformer-transducer-model.cc
  online-zipformer2-ctc-model-config.cc
  online-zipformer2-ctc-model.cc
  online-zipformer2-transducer-model.cc
  MNNUtils.cc
  packed-sequence.cc
  pad-sequence.cc
  parse-options.cc
  provider-config.cc
  provider.cc
  resample.cc
  session.cc
  silero-vad-model-config.cc
  silero-vad-model.cc
  slice.cc
  spoken-language-identification-impl.cc
  spoken-language-identification.cc
  stack.cc
  symbol-table.cc
  text-utils.cc
  transducer-keyword-decoder.cc
  transpose.cc
  unbind.cc
  utils.cc
  vad-model-config.cc
  vad-model.cc
  voice-activity-detector.cc
  wave-reader.cc
  wave-writer.cc
)

# speaker embedding extractor
list(APPEND sources
  speaker-embedding-extractor-impl.cc
  speaker-embedding-extractor-model.cc
  speaker-embedding-extractor-nemo-model.cc
  speaker-embedding-extractor.cc
  speaker-embedding-manager.cc
)

# audio tagging
list(APPEND sources
  audio-tagging-impl.cc
  audio-tagging-label-file.cc
  audio-tagging-model-config.cc
  audio-tagging.cc
  offline-ced-model.cc
  offline-zipformer-audio-tagging-model-config.cc
  offline-zipformer-audio-tagging-model.cc
)

# punctuation
list(APPEND sources
  offline-ct-transformer-model.cc
  offline-punctuation-impl.cc
  offline-punctuation-model-config.cc
  offline-punctuation.cc
  online-cnn-bilstm-model.cc
  online-punctuation-impl.cc
  online-punctuation-model-config.cc
  online-punctuation.cc
)

if(SHERPA_MNN_ENABLE_TTS)
  list(APPEND sources
    hifigan-vocoder.cc
    jieba-lexicon.cc
    kokoro-multi-lang-lexicon.cc
    lexicon.cc
    melo-tts-lexicon.cc
    offline-tts-character-frontend.cc
    offline-tts-frontend.cc
    offline-tts-impl.cc
    offline-tts-kokoro-model-config.cc
    offline-tts-kokoro-model.cc
    offline-tts-matcha-model-config.cc
    offline-tts-matcha-model.cc
    offline-tts-model-config.cc
    offline-tts-vits-model-config.cc
    offline-tts-vits-model.cc
    offline-tts.cc
    piper-phonemize-lexicon.cc
  )
endif()

list(APPEND sources
  offline-speech-denoiser-gtcrn-model-config.cc
  offline-speech-denoiser-gtcrn-model.cc
  offline-speech-denoiser-impl.cc
  offline-speech-denoiser-model-config.cc
  offline-speech-denoiser.cc
)

if(SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION)
  list(APPEND sources
    fast-clustering-config.cc
    fast-clustering.cc
    offline-speaker-diarization-impl.cc
    offline-speaker-diarization-result.cc
    offline-speaker-diarization.cc
    offline-speaker-segmentation-model-config.cc
    offline-speaker-segmentation-pyannote-model-config.cc
    offline-speaker-segmentation-pyannote-model.cc
  )
endif()

if(SHERPA_MNN_ENABLE_CHECK)
  list(APPEND sources log.cc)
endif()

# Always static build
add_library(sherpa-mnn-core STATIC ${sources})

set_target_properties(
    sherpa-mnn-core
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
)

if(APPLE)
  target_compile_options(sherpa-mnn-core PRIVATE
    -Wno-deprecated-declarations
  )
endif()

if(ANDROID_NDK)
  target_link_libraries(sherpa-mnn-core android log)
endif()
target_link_libraries(sherpa-mnn-core MNN)

target_link_libraries(sherpa-mnn-core
  kaldi-native-fbank-core
  kaldi-decoder-core
  ssentencepiece_core
)
if(DEFINED OHOS AND x${OHOS} STREQUAL xOHOS)
  target_link_libraries(sherpa-mnn-core
    hilog_ndk.z
    rawfile.z
  )
endif()


if(NOT WIN32)
  target_link_libraries(sherpa-mnn-core -lm)
endif()

if(NOT BUILD_SHARED_LIBS AND APPLE)
  target_link_libraries(sherpa-mnn-core "-framework Foundation")
endif()

target_link_libraries(sherpa-mnn-core fstfar fst)

if(SHERPA_MNN_ENABLE_TTS)
  target_link_libraries(sherpa-mnn-core piper_phonemize)
  target_link_libraries(sherpa-mnn-core cppjieba)
endif()

if(SHERPA_MNN_ENABLE_CHECK)
  target_compile_definitions(sherpa-mnn-core PUBLIC SHERPA_MNN_ENABLE_CHECK=1)

  if(SHERPA_MNN_HAVE_EXECINFO_H)
    target_compile_definitions(sherpa-mnn-core PRIVATE SHERPA_MNN_HAVE_EXECINFO_H=1)
  endif()

  if(SHERPA_MNN_HAVE_CXXABI_H)
    target_compile_definitions(sherpa-mnn-core PRIVATE SHERPA_MNN_HAVE_CXXABI_H=1)
  endif()
endif()

if(NOT BUILD_SHARED_LIBS AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  # This is for linux arm32 and arm64
  target_link_libraries(sherpa-mnn-core -ldl)
endif()

if(NOT WIN32 AND NOT SHERPA_MNN_ENABLE_WASM AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  target_link_libraries(sherpa-mnn-core -pthread)
endif()

if(SHERPA_MNN_ENABLE_BINARY)
  add_executable(sherpa-mnn sherpa-onnx.cc)
  add_executable(sherpa-mnn-keyword-spotter sherpa-onnx-keyword-spotter.cc)
  add_executable(sherpa-mnn-offline sherpa-onnx-offline.cc)
  add_executable(sherpa-mnn-offline-audio-tagging sherpa-onnx-offline-audio-tagging.cc)
  add_executable(sherpa-mnn-offline-language-identification sherpa-onnx-offline-language-identification.cc)
  add_executable(sherpa-mnn-offline-parallel sherpa-onnx-offline-parallel.cc)
  add_executable(sherpa-mnn-offline-punctuation sherpa-onnx-offline-punctuation.cc)
  add_executable(sherpa-mnn-online-punctuation sherpa-onnx-online-punctuation.cc)
  add_executable(sherpa-mnn-offline-denoiser sherpa-onnx-offline-denoiser.cc)

  if(SHERPA_MNN_ENABLE_TTS)
    add_executable(sherpa-mnn-offline-tts sherpa-onnx-offline-tts.cc)
  endif()

  if(SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION)
    add_executable(sherpa-mnn-offline-speaker-diarization sherpa-onnx-offline-speaker-diarization.cc)
  endif()

  set(main_exes
    sherpa-mnn
    sherpa-mnn-keyword-spotter
    sherpa-mnn-offline
    sherpa-mnn-offline-audio-tagging
    sherpa-mnn-offline-language-identification
    sherpa-mnn-offline-parallel
    sherpa-mnn-offline-punctuation
    sherpa-mnn-offline-denoiser
    sherpa-mnn-online-punctuation
  )
  if(SHERPA_MNN_ENABLE_TTS)
    list(APPEND main_exes
      sherpa-mnn-offline-tts
    )
  endif()

  if(SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION)
    list(APPEND main_exes
      sherpa-mnn-offline-speaker-diarization
    )
  endif()

  foreach(exe IN LISTS main_exes)
    target_link_libraries(${exe} sherpa-mnn-core)
  endforeach()

  if(NOT WIN32)
    foreach(exe IN LISTS main_exes)
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib")
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

      if(SHERPA_MNN_ENABLE_PYTHON)
        target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      endif()
    endforeach()
  endif()
endif()

if(NOT BUILD_SHARED_LIBS)
  install(TARGETS sherpa-mnn-core DESTINATION lib)
endif()

if(SHERPA_MNN_ENABLE_BINARY)
  install(
    TARGETS
      ${main_exes}
    DESTINATION
      bin
  )
endif()

if(SHERPA_MNN_HAS_ALSA AND SHERPA_MNN_ENABLE_BINARY)
  add_executable(sherpa-mnn-alsa sherpa-onnx-alsa.cc alsa.cc)
  add_executable(sherpa-mnn-alsa-offline sherpa-onnx-alsa-offline.cc alsa.cc)
  add_executable(sherpa-mnn-alsa-offline-audio-tagging sherpa-onnx-alsa-offline-audio-tagging.cc alsa.cc)
  add_executable(sherpa-mnn-alsa-offline-speaker-identification sherpa-onnx-alsa-offline-speaker-identification.cc alsa.cc)
  add_executable(sherpa-mnn-keyword-spotter-alsa sherpa-onnx-keyword-spotter-alsa.cc alsa.cc)
  add_executable(sherpa-mnn-vad-alsa sherpa-onnx-vad-alsa.cc alsa.cc)


  if(SHERPA_MNN_ENABLE_TTS)
    add_executable(sherpa-mnn-offline-tts-play-alsa sherpa-onnx-offline-tts-play-alsa.cc alsa-play.cc)
  endif()

  set(exes
    sherpa-mnn-alsa
    sherpa-mnn-alsa-offline
    sherpa-mnn-alsa-offline-speaker-identification
    sherpa-mnn-keyword-spotter-alsa
    sherpa-mnn-vad-alsa
    sherpa-mnn-alsa-offline-audio-tagging
  )

  if(SHERPA_MNN_ENABLE_TTS)
    list(APPEND exes
      sherpa-mnn-offline-tts-play-alsa
    )
  endif()

  #   # To fix the following error for Windows when building exe
  #   #  mismatch detected for 'RuntimeLibrary': value 'MT_StaticRelease' doesn't match value 'MD_Dynamic Release'

  foreach(exe IN LISTS exes)
    target_link_libraries(${exe} sherpa-mnn-core)
  endforeach()

  foreach(exe IN LISTS exes)
    if(DEFINED ENV{SHERPA_MNN_ALSA_LIB_DIR})
      target_link_libraries(${exe} -L$ENV{SHERPA_MNN_ALSA_LIB_DIR} -lasound)
    else()
      target_link_libraries(${exe} asound)
    endif()
  endforeach()

  if(NOT WIN32)
    foreach(exe IN LISTS exes)
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib")
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../../../sherpa_onnx/lib")
    endforeach()

    if(SHERPA_MNN_ENABLE_PYTHON)
      foreach(exe IN LISTS exes)
        target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      endforeach()
    endif()
  endif()

  install(
    TARGETS ${exes}
    DESTINATION
      bin
  )
endif()

if(SHERPA_MNN_ENABLE_PORTAUDIO AND SHERPA_MNN_ENABLE_BINARY)
  if(SHERPA_MNN_ENABLE_TTS)
    add_executable(sherpa-mnn-offline-tts-play
      sherpa-onnx-offline-tts-play.cc
      microphone.cc
    )
  endif()

  add_executable(sherpa-mnn-keyword-spotter-microphone
    sherpa-onnx-keyword-spotter-microphone.cc
    microphone.cc
  )

  add_executable(sherpa-mnn-microphone
    sherpa-onnx-microphone.cc
    microphone.cc
  )

  add_executable(sherpa-mnn-microphone-offline
    sherpa-onnx-microphone-offline.cc
    microphone.cc
  )

  add_executable(sherpa-mnn-vad-microphone
    sherpa-onnx-vad-microphone.cc
    microphone.cc
  )

  add_executable(sherpa-mnn-vad-with-offline-asr
    sherpa-onnx-vad-with-offline-asr.cc
  )

  add_executable(sherpa-mnn-vad-microphone-offline-asr
    sherpa-onnx-vad-microphone-offline-asr.cc
    microphone.cc
  )

  add_executable(sherpa-mnn-microphone-offline-speaker-identification
    sherpa-onnx-microphone-offline-speaker-identification.cc
    microphone.cc
  )

  add_executable(sherpa-mnn-microphone-offline-audio-tagging
    sherpa-onnx-microphone-offline-audio-tagging.cc
    microphone.cc
  )

  set(exes
    sherpa-mnn-microphone
    sherpa-mnn-keyword-spotter-microphone
    sherpa-mnn-microphone-offline
    sherpa-mnn-microphone-offline-speaker-identification
    sherpa-mnn-microphone-offline-audio-tagging
    sherpa-mnn-vad-microphone
    sherpa-mnn-vad-microphone-offline-asr
    sherpa-mnn-vad-with-offline-asr
  )
  if(SHERPA_MNN_ENABLE_TTS)
    list(APPEND exes
      sherpa-mnn-offline-tts-play
    )
  endif()

  foreach(exe IN LISTS exes)
    target_link_libraries(${exe} portaudio_static sherpa-mnn-core)
  endforeach()

  if(NOT WIN32)
    foreach(exe IN LISTS exes)
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib")
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../../../sherpa_onnx/lib")
    endforeach()

    if(SHERPA_MNN_ENABLE_PYTHON)
      foreach(exe IN LISTS exes)
        target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      endforeach()
    endif()
  endif()

  install(
    TARGETS ${exes}
    DESTINATION
      bin
  )
endif()

if(SHERPA_MNN_ENABLE_WEBSOCKET AND SHERPA_MNN_ENABLE_BINARY)
  add_definitions(-DASIO_STANDALONE)
  add_definitions(-D_WEBSOCKETPP_CPP11_STL_)

  add_executable(sherpa-mnn-online-websocket-server
    online-websocket-server-impl.cc
    online-websocket-server.cc
  )
  target_link_libraries(sherpa-mnn-online-websocket-server sherpa-mnn-core)

  add_executable(sherpa-mnn-online-websocket-client
    online-websocket-client.cc
  )
  target_link_libraries(sherpa-mnn-online-websocket-client sherpa-mnn-core)

  if(NOT WIN32)
    target_compile_options(sherpa-mnn-online-websocket-server PRIVATE -Wno-deprecated-declarations)

    target_compile_options(sherpa-mnn-online-websocket-client PRIVATE -Wno-deprecated-declarations)
  endif()

  # For offline websocket
  add_executable(sherpa-mnn-offline-websocket-server
    offline-websocket-server-impl.cc
    offline-websocket-server.cc
  )
  target_link_libraries(sherpa-mnn-offline-websocket-server sherpa-mnn-core)

  if(NOT WIN32)
    target_compile_options(sherpa-mnn-offline-websocket-server PRIVATE -Wno-deprecated-declarations)
  endif()

  if(NOT WIN32)
    target_link_libraries(sherpa-mnn-online-websocket-server "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib")
    target_link_libraries(sherpa-mnn-online-websocket-server "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

    target_link_libraries(sherpa-mnn-online-websocket-client "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib")
    target_link_libraries(sherpa-mnn-online-websocket-client "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

    target_link_libraries(sherpa-mnn-offline-websocket-server "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib")
    target_link_libraries(sherpa-mnn-offline-websocket-server "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

    if(SHERPA_MNN_ENABLE_PYTHON AND NOT WIN32)
      target_link_libraries(sherpa-mnn-online-websocket-server "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      target_link_libraries(sherpa-mnn-online-websocket-client "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      target_link_libraries(sherpa-mnn-offline-websocket-server "-Wl,-rpath,${SHERPA_MNN_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
    endif()
  endif()

  install(
    TARGETS
      sherpa-mnn-online-websocket-server
      sherpa-mnn-online-websocket-client
      sherpa-mnn-offline-websocket-server
    DESTINATION
      bin
  )
endif()

if(SHERPA_MNN_ENABLE_TESTS)
  set(sherpa_onnx_test_srcs
    cat-test.cc
    circular-buffer-test.cc
    context-graph-test.cc
    packed-sequence-test.cc
    pad-sequence-test.cc
    regex-lang-test.cc
    slice-test.cc
    stack-test.cc
    text-utils-test.cc
    text2token-test.cc
    transpose-test.cc
    unbind-test.cc
    utfcpp-test.cc
  )
  if(SHERPA_MNN_ENABLE_TTS)
    list(APPEND sherpa_onnx_test_srcs
      cppjieba-test.cc
      piper-phonemize-test.cc
    )
  endif()

  if(SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION)
    list(APPEND sherpa_onnx_test_srcs
      fast-clustering-test.cc
    )
  endif()

  list(APPEND sherpa_onnx_test_srcs
    speaker-embedding-manager-test.cc
  )

  function(sherpa_onnx_add_test source)
    get_filename_component(name ${source} NAME_WE)
    set(target_name ${name})
    add_executable(${target_name} "${source}")

    target_link_libraries(${target_name}
      PRIVATE
        gtest
        gtest_main
        sherpa-mnn-core
    )

    add_test(NAME "${target_name}"
      COMMAND
        $<TARGET_FILE:${target_name}>
    )
  endfunction()

  foreach(source IN LISTS sherpa_onnx_test_srcs)
    sherpa_onnx_add_test(${source})
  endforeach()
endif()

set(srcs_to_check)
foreach(s IN LISTS sources)
  list(APPEND srcs_to_check ${CMAKE_CURRENT_LIST_DIR}/${s})
endforeach()

# For clang-tidy
add_custom_target(
  clang-tidy-check
  clang-tidy -p ${CMAKE_BINARY_DIR}/compile_commands.json --config-file ${CMAKE_SOURCE_DIR}/.clang-tidy ${srcs_to_check}
  DEPENDS ${sources})

add_custom_target(check DEPENDS clang-tidy-check)
