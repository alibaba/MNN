function(download_piper_phonemize)
  include(FetchContent)

  set(piper_phonemize_URL  "https://github.com/csukuangfj/piper-phonemize/archive/78a788e0b719013401572d70fef372e77bff8e43.zip")
  set(piper_phonemize_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/piper-phonemize-78a788e0b719013401572d70fef372e77bff8e43.zip")
  set(piper_phonemize_HASH "SHA256=89641a46489a4898754643ce57bda9c9b54b4ca46485fdc02bf0dc84b866645d")

  # If you don't have access to the Internet,
  # please pre-download kaldi-decoder
  set(possible_file_locations
    $ENV{HOME}/Downloads/piper-phonemize-78a788e0b719013401572d70fef372e77bff8e43.zip
    ${CMAKE_SOURCE_DIR}/piper-phonemize-78a788e0b719013401572d70fef372e77bff8e43.zip
    ${CMAKE_BINARY_DIR}/piper-phonemize-78a788e0b719013401572d70fef372e77bff8e43.zip
    /tmp/piper-phonemize-78a788e0b719013401572d70fef372e77bff8e43.zip
    /star-fj/fangjun/download/github/piper-phonemize-78a788e0b719013401572d70fef372e77bff8e43.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(piper_phonemize_URL  "${f}")
      file(TO_CMAKE_PATH "${piper_phonemize_URL}" piper_phonemize_URL)
      message(STATUS "Found local downloaded espeak-ng: ${piper_phonemize_URL}")
      set(piper_phonemize_URL2 )
      break()
    endif()
  endforeach()

  FetchContent_Declare(piper_phonemize
    URL
      ${piper_phonemize_URL}
      ${piper_phonemize_URL2}
    URL_HASH          ${piper_phonemize_HASH}
  )

  FetchContent_GetProperties(piper_phonemize)
  if(NOT piper_phonemize_POPULATED)
    message(STATUS "Downloading piper-phonemize from ${piper_phonemize_URL}")
    FetchContent_Populate(piper_phonemize)
  endif()
  message(STATUS "piper-phonemize is downloaded to ${piper_phonemize_SOURCE_DIR}")
  message(STATUS "piper-phonemize binary dir is ${piper_phonemize_BINARY_DIR}")

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${piper_phonemize_SOURCE_DIR} ${piper_phonemize_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(_build_shared_libs_bak)
    set_target_properties(piper_phonemize
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  if(WIN32 AND MSVC)
    target_compile_options(piper_phonemize PUBLIC
      /wd4309
    )
  endif()

  target_include_directories(piper_phonemize
    INTERFACE
      ${piper_phonemize_SOURCE_DIR}/src/include
  )

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS
      piper_phonemize
    DESTINATION lib)
  endif()
endfunction()

download_piper_phonemize()
