include_directories(${CMAKE_SOURCE_DIR})

if(NOT DEFINED ANDROID_ABI)
  if(NOT DEFINED ENV{JAVA_HOME})
    message(FATAL_ERROR "Please set the environment variable JAVA_HOME")
  endif()
  include_directories($ENV{JAVA_HOME}/include)
  include_directories($ENV{JAVA_HOME}/include/linux)
  include_directories($ENV{JAVA_HOME}/include/darwin)
  include_directories($ENV{JAVA_HOME}/include/win32)
endif()

set(sources
  audio-tagging.cc
  jni.cc
  keyword-spotter.cc
  offline-punctuation.cc
  offline-recognizer.cc
  offline-stream.cc
  online-punctuation.cc
  online-recognizer.cc
  online-stream.cc
  speaker-embedding-extractor.cc
  speaker-embedding-manager.cc
  spoken-language-identification.cc
  voice-activity-detector.cc
  wave-reader.cc
  wave-writer.cc
)

if(SHERPA_MNN_ENABLE_TTS)
  list(APPEND sources
    offline-tts.cc
  )
endif()

if(SHERPA_MNN_ENABLE_SPEAKER_DIARIZATION)
  list(APPEND sources
    offline-speaker-diarization.cc
  )
endif()

add_library(sherpa-mnn-jni SHARED ${sources})

target_compile_definitions(sherpa-mnn-jni PRIVATE SHERPA_MNN_BUILD_SHARED_LIBS=1)
target_compile_definitions(sherpa-mnn-jni PRIVATE SHERPA_MNN_BUILD_MAIN_LIB=1)

target_link_libraries(sherpa-mnn-jni sherpa-mnn-core)
install(TARGETS sherpa-mnn-jni DESTINATION lib)
