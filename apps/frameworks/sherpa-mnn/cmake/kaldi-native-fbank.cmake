function(download_kaldi_native_fbank)
  include(FetchContent)

  set(kaldi_native_fbank_URL   "https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.21.1.tar.gz")
  set(kaldi_native_fbank_URL2  "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/kaldi-native-fbank-1.21.1.tar.gz")
  set(kaldi_native_fbank_HASH "SHA256=37c1aa230b00fe062791d800d8fc50aa3de215918d3dce6440699e67275d859e")

  set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download kaldi-native-fbank
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-native-fbank-1.21.1.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldi-native-fbank-1.21.1.tar.gz
    ${CMAKE_BINARY_DIR}/kaldi-native-fbank-1.21.1.tar.gz
    /tmp/kaldi-native-fbank-1.21.1.tar.gz
    /star-fj/fangjun/download/github/kaldi-native-fbank-1.21.1.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_native_fbank_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldi_native_fbank_URL}" kaldi_native_fbank_URL)
      message(STATUS "Found local downloaded kaldi-native-fbank: ${kaldi_native_fbank_URL}")
      set(kaldi_native_fbank_URL2 )
      break()
    endif()
  endforeach()

  FetchContent_Declare(kaldi_native_fbank
    URL
      ${kaldi_native_fbank_URL}
      ${kaldi_native_fbank_URL2}
    URL_HASH          ${kaldi_native_fbank_HASH}
  )

  FetchContent_GetProperties(kaldi_native_fbank)
  if(NOT kaldi_native_fbank_POPULATED)
    message(STATUS "Downloading kaldi-native-fbank from ${kaldi_native_fbank_URL}")
    FetchContent_Populate(kaldi_native_fbank)
  endif()
  message(STATUS "kaldi-native-fbank is downloaded to ${kaldi_native_fbank_SOURCE_DIR}")
  message(STATUS "kaldi-native-fbank's binary dir is ${kaldi_native_fbank_BINARY_DIR}")

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${kaldi_native_fbank_SOURCE_DIR} ${kaldi_native_fbank_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(_build_shared_libs_bak)
    set_target_properties(kaldi-native-fbank-core
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  target_include_directories(kaldi-native-fbank-core
    INTERFACE
      ${kaldi_native_fbank_SOURCE_DIR}/
  )

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS kaldi-native-fbank-core DESTINATION lib)
  endif()
endfunction()

download_kaldi_native_fbank()
