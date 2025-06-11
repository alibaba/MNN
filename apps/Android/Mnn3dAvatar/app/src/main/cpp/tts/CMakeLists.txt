
cmake_minimum_required(VERSION 3.13)

project(mnn_tts_SDK VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_BERTVITS2 "Build BertVit2 TTS " ON)
option(BUILD_PIPER "Build PIPER TTS " OFF)

include_directories(
	${CMAKE_CURRENT_LIST_DIR}/include
	${CMAKE_CURRENT_LIST_DIR}/third_party/include/nlohmann
)

if(BUILD_BERTVITS2)
  include_directories(
  	${CMAKE_CURRENT_LIST_DIR}/include/bertvits2
  )
endif()

if(BUILD_PIPER)
  include_directories(
  	${CMAKE_CURRENT_LIST_DIR}/include/piper
    ${CMAKE_CURRENT_LIST_DIR}/third_party/piper/espeak-ng/src/include/espeak-ng/
  )
endif()

set(SHARED_SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/src/mnn_tts_config.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/mnn_tts_sdk.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/mnn_tts_logger.cpp
)

set(BERTVITS2_SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/an_to_cn.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/mnn_bertvits2_tts_impl.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/text_preprocessor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/pinyin.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/chinese_g2p.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/chinese_bert.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/english_bert.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/english_g2p.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/tone_adjuster.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/tts_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/bertvits2/word_spliter.cpp
)
if(BUILD_PIPER)
set(PIPER_SOURCE_FILES
  ${CMAKE_CURRENT_LIST_DIR}/src/piper/audio_generator.cpp 
  ${CMAKE_CURRENT_LIST_DIR}/src/piper/mnn_piper_tts_impl.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/piper/phoneme_ids.cpp
)
endif()

if(BUILD_PIPER)
  add_subdirectory(third_party/piper/espeak-ng)
endif()

add_library(${PROJECT_NAME} SHARED ${PIPER_SOURCE_FILES} ${BERTVITS2_SOURCE_FILES} ${SHARED_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} log MNN )
