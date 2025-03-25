IF(MNN_BUILD_AUDIO)
  # imgproc submodules start
  option(MNN_AUDIO_TEST "Enable audio test" OFF)

  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/imageHelper/)

  # include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
  if(${MNN_AUDIO_TEST})
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
  endif()

  # include dir
  include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

  # source files
  FILE(GLOB AUDIO_SRCS ${CMAKE_CURRENT_LIST_DIR}/source/*.cpp)

  IF(MNN_SEP_BUILD)
    IF(MNN_BUILD_SHARED_LIBS)
      add_library(MNNAudio SHARED ${AUDIO_SRCS})
      target_link_libraries(MNNAudio MNN MNN_Express)
    ELSE()
      add_library(MNNAudio STATIC ${AUDIO_SRCS})
    ENDIF()
  ELSE()
    add_library(MNNAudio OBJECT ${AUDIO_SRCS})
  ENDIF()
  # copy header files
  IF(CMAKE_SYSTEM_NAME MATCHES "^Android" AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    IF(NOT NATIVE_INCLUDE_OUTPUT)
      set(NATIVE_INCLUDE_OUTPUT ".")
    ENDIF()
    add_custom_command(
      TARGET MNNAudio
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/include ${NATIVE_INCLUDE_OUTPUT}
    )
  ELSE()
    INSTALL(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/audio DESTINATION include FILES_MATCHING PATTERN *.hpp)
  ENDIF()
ENDIF()