function(download_kaldi_decoder)
  include(FetchContent)

  set(kaldi_decoder_URL  "https://github.com/k2-fsa/kaldi-decoder/archive/refs/tags/v0.2.6.tar.gz")
  set(kaldi_decoder_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/kaldi-decoder-0.2.6.tar.gz")
  set(kaldi_decoder_HASH "SHA256=b13c78b37495cafc6ef3f8a7b661b349c55a51abbd7f7f42f389408dcf86a463")

  set(KALDI_DECODER_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_DECODER_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDIFST_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download kaldi-decoder
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-decoder-0.2.6.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldi-decoder-0.2.6.tar.gz
    ${CMAKE_BINARY_DIR}/kaldi-decoder-0.2.6.tar.gz
    /tmp/kaldi-decoder-0.2.6.tar.gz
    /star-fj/fangjun/download/github/kaldi-decoder-0.2.6.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_decoder_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldi_decoder_URL}" kaldi_decoder_URL)
      message(STATUS "Found local downloaded kaldi-decoder: ${kaldi_decoder_URL}")
      set(kaldi_decoder_URL2 )
      break()
    endif()
  endforeach()

  FetchContent_Declare(kaldi_decoder
    URL
      ${kaldi_decoder_URL}
      ${kaldi_decoder_URL2}
    URL_HASH          ${kaldi_decoder_HASH}
  )

  FetchContent_GetProperties(kaldi_decoder)
  if(NOT kaldi_decoder_POPULATED)
    message(STATUS "Downloading kaldi-decoder from ${kaldi_decoder_URL}")
    FetchContent_Populate(kaldi_decoder)
  endif()
  message(STATUS "kaldi-decoder is downloaded to ${kaldi_decoder_SOURCE_DIR}")
  message(STATUS "kaldi-decoder's binary dir is ${kaldi_decoder_BINARY_DIR}")

  include_directories(${kaldi_decoder_SOURCE_DIR})

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${kaldi_decoder_SOURCE_DIR} ${kaldi_decoder_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(_build_shared_libs_bak)
    set_target_properties(
        kaldi-decoder-core
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  if(WIN32 AND MSVC)
    target_compile_options(kaldi-decoder-core PUBLIC
      /wd4018
      /wd4291
    )
  endif()

  target_include_directories(kaldi-decoder-core
    INTERFACE
      ${kaldi-decoder_SOURCE_DIR}/
  )
  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS
      kaldi-decoder-core
      kaldifst_core
      fst
      fstfar
    DESTINATION lib)
  endif()
endfunction()

download_kaldi_decoder()

