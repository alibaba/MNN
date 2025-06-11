# Copyright (c)  2020  Xiaomi Corporation (author: Fangjun Kuang)

function(download_openfst)
  include(FetchContent)

  set(openfst_URL  "https://github.com/csukuangfj/openfst/archive/refs/tags/sherpa-onnx-2024-06-19.tar.gz")
  set(openfst_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/openfst-sherpa-onnx-2024-06-19.tar.gz")
  set(openfst_HASH "SHA256=5c98e82cc509c5618502dde4860b8ea04d843850ed57e6d6b590b644b268853d")

  # If you don't have access to the Internet,
  # please pre-download it
  set(possible_file_locations
    $ENV{HOME}/Downloads/openfst-sherpa-onnx-2024-06-19.tar.gz
    ${CMAKE_SOURCE_DIR}/openfst-sherpa-onnx-2024-06-19.tar.gz
    ${CMAKE_BINARY_DIR}/openfst-sherpa-onnx-2024-06-19.tar.gz
    /tmp/openfst-sherpa-onnx-2024-06-19.tar.gz
    /star-fj/fangjun/download/github/openfst-sherpa-onnx-2024-06-19.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(openfst_URL  "${f}")
      file(TO_CMAKE_PATH "${openfst_URL}" openfst_URL)
      set(openfst_URL2)
      break()
    endif()
  endforeach()

  set(HAVE_BIN OFF CACHE BOOL "" FORCE)
  set(HAVE_SCRIPT OFF CACHE BOOL "" FORCE)
  set(HAVE_COMPACT OFF CACHE BOOL "" FORCE)
  set(HAVE_COMPRESS OFF CACHE BOOL "" FORCE)
  set(HAVE_CONST OFF CACHE BOOL "" FORCE)
  set(HAVE_FAR ON CACHE BOOL "" FORCE)
  set(HAVE_GRM OFF CACHE BOOL "" FORCE)
  set(HAVE_PDT OFF CACHE BOOL "" FORCE)
  set(HAVE_MPDT OFF CACHE BOOL "" FORCE)
  set(HAVE_LINEAR OFF CACHE BOOL "" FORCE)
  set(HAVE_LOOKAHEAD OFF CACHE BOOL "" FORCE)
  set(HAVE_NGRAM OFF CACHE BOOL "" FORCE)
  set(HAVE_PYTHON OFF CACHE BOOL "" FORCE)
  set(HAVE_SPECIAL OFF CACHE BOOL "" FORCE)

  if(NOT WIN32)
    FetchContent_Declare(openfst
      URL
        ${openfst_URL}
        ${openfst_URL2}
      URL_HASH          ${openfst_HASH}
      PATCH_COMMAND
        sed -i.bak s/enable_testing\(\)//g "src/CMakeLists.txt" &&
        sed -i.bak s/add_subdirectory\(test\)//g "src/CMakeLists.txt" &&
        sed -i.bak /message/d "src/script/CMakeLists.txt"
        # sed -i.bak s/add_subdirectory\(script\)//g "src/CMakeLists.txt" &&
        # sed -i.bak s/add_subdirectory\(extensions\)//g "src/CMakeLists.txt"
    )
  else()
    FetchContent_Declare(openfst
      URL               ${openfst_URL}
      URL_HASH          ${openfst_HASH}
    )
  endif()

  FetchContent_GetProperties(openfst)
  if(NOT openfst_POPULATED)
    message(STATUS "Downloading openfst from ${openfst_URL}")
    FetchContent_Populate(openfst)
  endif()
  message(STATUS "openfst is downloaded to ${openfst_SOURCE_DIR}")

  if(_build_shared_libs_bak)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${openfst_SOURCE_DIR} ${openfst_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(_build_shared_libs_bak)
    set_target_properties(fst fstfar
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  set(openfst_SOURCE_DIR ${openfst_SOURCE_DIR} PARENT_SCOPE)

  set_target_properties(fst PROPERTIES OUTPUT_NAME "sherpa-mnn-fst")
  set_target_properties(fstfar PROPERTIES OUTPUT_NAME "sherpa-mnn-fstfar")

  if(LINUX)
    target_compile_options(fst PUBLIC -Wno-missing-template-keyword)
  endif()

  target_include_directories(fst
    PUBLIC
      ${openfst_SOURCE_DIR}/src/include
  )

  target_include_directories(fstfar
    PUBLIC
      ${openfst_SOURCE_DIR}/src/include
  )
  # installed in ./kaldi-decoder.cmake
endfunction()

download_openfst()
