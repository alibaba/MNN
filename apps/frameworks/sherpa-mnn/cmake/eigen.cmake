function(download_eigen)
  include(FetchContent)

  set(eigen_URL  "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz")
  set(eigen_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/eigen-3.4.0.tar.gz")
  set(eigen_HASH "SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72")

  # If you don't have access to the Internet,
  # please pre-download eigen
  set(possible_file_locations
    $ENV{HOME}/Downloads/eigen-3.4.0.tar.gz
    ${CMAKE_SOURCE_DIR}/eigen-3.4.0.tar.gz
    ${CMAKE_BINARY_DIR}/eigen-3.4.0.tar.gz
    /tmp/eigen-3.4.0.tar.gz
    /star-fj/fangjun/download/github/eigen-3.4.0.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(eigen_URL  "${f}")
      file(TO_CMAKE_PATH "${eigen_URL}" eigen_URL)
      message(STATUS "Found local downloaded eigen: ${eigen_URL}")
      set(eigen_URL2)
      break()
    endif()
  endforeach()

  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(eigen
    URL               ${eigen_URL}
    URL_HASH          ${eigen_HASH}
  )

  FetchContent_GetProperties(eigen)
  if(NOT eigen_POPULATED)
    message(STATUS "Downloading eigen from ${eigen_URL}")
    FetchContent_Populate(eigen)
  endif()
  message(STATUS "eigen is downloaded to ${eigen_SOURCE_DIR}")
  message(STATUS "eigen's binary dir is ${eigen_BINARY_DIR}")

  add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_eigen()

