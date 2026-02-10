function(download_espeak_ng_for_piper)
  include(FetchContent)

  set(espeak_ng_URL  "https://github.com/csukuangfj/espeak-ng/archive/f6fed6c58b5e0998b8e68c6610125e2d07d595a7.zip")
  set(espeak_ng_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/espeak-ng-f6fed6c58b5e0998b8e68c6610125e2d07d595a7.zip")
  set(espeak_ng_HASH "SHA256=70cbf4050e7a014aae19140b05e57249da4720f56128459fbe3a93beaf971ae6")

  set(BUILD_ESPEAK_NG_TESTS OFF CACHE BOOL "" FORCE)
  set(USE_ASYNC OFF CACHE BOOL "" FORCE)
  set(USE_MBROLA OFF CACHE BOOL "" FORCE)
  set(USE_LIBSONIC OFF CACHE BOOL "" FORCE)
  set(USE_LIBPCAUDIO OFF CACHE BOOL "" FORCE)
  set(USE_KLATT OFF CACHE BOOL "" FORCE)
  set(USE_SPEECHPLAYER OFF CACHE BOOL "" FORCE)
  set(EXTRA_cmn ON CACHE BOOL "" FORCE)
  set(EXTRA_ru ON CACHE BOOL "" FORCE)
  if (NOT SHERPA_ONNX_ENABLE_EPSEAK_NG_EXE)
    set(BUILD_ESPEAK_NG_EXE OFF CACHE BOOL "" FORCE)
  endif()

  # If you don't have access to the Internet,
  # please pre-download kaldi-decoder
  set(possible_file_locations
    $ENV{HOME}/Downloads/espeak-ng-f6fed6c58b5e0998b8e68c6610125e2d07d595a7.zip
    ${CMAKE_SOURCE_DIR}/espeak-ng-f6fed6c58b5e0998b8e68c6610125e2d07d595a7.zip
    ${CMAKE_BINARY_DIR}/espeak-ng-f6fed6c58b5e0998b8e68c6610125e2d07d595a7.zip
    /tmp/espeak-ng-f6fed6c58b5e0998b8e68c6610125e2d07d595a7.zip
    /star-fj/fangjun/download/github/espeak-ng-f6fed6c58b5e0998b8e68c6610125e2d07d595a7.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(espeak_ng_URL  "${f}")
      file(TO_CMAKE_PATH "${espeak_ng_URL}" espeak_ng_URL)
      message(STATUS "Found local downloaded espeak-ng: ${espeak_ng_URL}")
      set(espeak_ng_URL2 )
      break()
    endif()
  endforeach()

  FetchContent_Declare(espeak_ng
    URL
      ${espeak_ng_URL}
      ${espeak_ng_URL2}
    URL_HASH          ${espeak_ng_HASH}
  )

  FetchContent_GetProperties(espeak_ng)
  if(NOT espeak_ng_POPULATED)
    message(STATUS "Downloading espeak-ng from ${espeak_ng_URL}")
    FetchContent_Populate(espeak_ng)
  endif()
  message(STATUS "espeak-ng is downloaded to ${espeak_ng_SOURCE_DIR}")
  message(STATUS "espeak-ng binary dir is ${espeak_ng_BINARY_DIR}")

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${espeak_ng_SOURCE_DIR} ${espeak_ng_BINARY_DIR})

  if(_build_shared_libs_bak)
    set_target_properties(espeak-ng
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  set(espeak_ng_SOURCE_DIR ${espeak_ng_SOURCE_DIR} PARENT_SCOPE)

  if(WIN32 AND MSVC)
    target_compile_options(ucd PUBLIC
      /wd4309
    )

    target_compile_options(espeak-ng PUBLIC
      /wd4005
      /wd4018
      /wd4067
      /wd4068
      /wd4090
      /wd4101
      /wd4244
      /wd4267
      /wd4996
    )

    if(TARGET espeak-ng-bin)
      target_compile_options(espeak-ng-bin PRIVATE
        /wd4244
        /wd4024
        /wd4047
        /wd4067
        /wd4267
        /wd4996
      )
    endif()
  endif()

  if(UNIX AND NOT APPLE)
    target_compile_options(espeak-ng PRIVATE
      -Wno-unused-result
      -Wno-format-overflow
      -Wno-format-truncation
      -Wno-uninitialized
      -Wno-format
    )

    if(TARGET espeak-ng-bin)
      target_compile_options(espeak-ng-bin PRIVATE
        -Wno-unused-result
      )
    endif()
  endif()

  target_include_directories(espeak-ng
    INTERFACE
      ${espeak_ng_SOURCE_DIR}/src/include
      ${espeak_ng_SOURCE_DIR}/src/ucd-tools/src/include
  )

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS
      espeak-ng
      ucd
    DESTINATION lib)
  endif()
endfunction()

download_espeak_ng_for_piper()
