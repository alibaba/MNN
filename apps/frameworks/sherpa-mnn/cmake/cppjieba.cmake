function(download_cppjieba)
  include(FetchContent)

  set(cppjieba_URL  "https://github.com/csukuangfj/cppjieba/archive/refs/tags/sherpa-onnx-2024-04-19.tar.gz")
  set(cppjieba_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/cppjieba-sherpa-onnx-2024-04-19.tar.gz")
  set(cppjieba_HASH "SHA256=03e5264687f0efaef05487a07d49c3f4c0f743347bfbf825df4b30cc75ac5288")

  # If you don't have access to the Internet,
  # please pre-download cppjieba
  set(possible_file_locations
    $ENV{HOME}/Downloads/cppjieba-sherpa-onnx-2024-04-19.tar.gz
    ${CMAKE_SOURCE_DIR}/cppjieba-sherpa-onnx-2024-04-19.tar.gz
    ${CMAKE_BINARY_DIR}/cppjieba-sherpa-onnx-2024-04-19.tar.gz
    /tmp/cppjieba-sherpa-onnx-2024-04-19.tar.gz
    /star-fj/fangjun/download/github/cppjieba-sherpa-onnx-2024-04-19.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(cppjieba_URL  "${f}")
      file(TO_CMAKE_PATH "${cppjieba_URL}" cppjieba_URL)
      message(STATUS "Found local downloaded cppjieba: ${cppjieba_URL}")
      set(cppjieba_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(cppjieba
    URL
      ${cppjieba_URL}
      ${cppjieba_URL2}
    URL_HASH
      ${cppjieba_HASH}
  )

  FetchContent_GetProperties(cppjieba)
  if(NOT cppjieba_POPULATED)
    message(STATUS "Downloading cppjieba ${cppjieba_URL}")
    FetchContent_Populate(cppjieba)
  endif()
  message(STATUS "cppjieba is downloaded to ${cppjieba_SOURCE_DIR}")
  add_subdirectory(${cppjieba_SOURCE_DIR} ${cppjieba_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_cppjieba()
