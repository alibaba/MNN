function(download_kaldifst)
  include(FetchContent)

  set(kaldifst_URL  "https://github.com/k2-fsa/kaldifst/archive/refs/tags/v1.7.11.tar.gz")
  set(kaldifst_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/kaldifst-1.7.11.tar.gz")
  set(kaldifst_HASH "SHA256=b43b3332faa2961edc730e47995a58cd4e22ead21905d55b0c4a41375b4a525f")

  # If you don't have access to the Internet,
  # please pre-download kaldifst
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldifst-1.7.11.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldifst-1.7.11.tar.gz
    ${CMAKE_BINARY_DIR}/kaldifst-1.7.11.tar.gz
    /tmp/kaldifst-1.7.11.tar.gz
    /star-fj/fangjun/download/github/kaldifst-1.7.11.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldifst_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldifst_URL}" kaldifst_URL)
      message(STATUS "Found local downloaded kaldifst: ${kaldifst_URL}")
      set(kaldifst_URL2)
      break()
    endif()
  endforeach()

  set(KALDIFST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDIFST_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kaldifst
    URL               ${kaldifst_URL}
    URL_HASH          ${kaldifst_HASH}
  )

  FetchContent_GetProperties(kaldifst)
  if(NOT kaldifst_POPULATED)
    message(STATUS "Downloading kaldifst from ${kaldifst_URL}")
    FetchContent_Populate(kaldifst)
  endif()
  message(STATUS "kaldifst is downloaded to ${kaldifst_SOURCE_DIR}")
  message(STATUS "kaldifst's binary dir is ${kaldifst_BINARY_DIR}")

  list(APPEND CMAKE_MODULE_PATH ${kaldifst_SOURCE_DIR}/cmake)

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${kaldifst_SOURCE_DIR} ${kaldifst_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(_build_shared_libs_bak)
    set_target_properties(kaldifst_core
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  target_include_directories(kaldifst_core
    PUBLIC
      ${kaldifst_SOURCE_DIR}/
  )

  set_target_properties(kaldifst_core PROPERTIES OUTPUT_NAME "sherpa-mnn-kaldifst-core")
  # installed in ./kaldi-decoder.cmake
endfunction()

download_kaldifst()
