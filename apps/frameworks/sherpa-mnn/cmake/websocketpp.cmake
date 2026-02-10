function(download_websocketpp)
  include(FetchContent)

  # The latest commit on the develop branch os as 2022-10-22
  set(websocketpp_URL  "https://github.com/zaphoyd/websocketpp/archive/b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip")
  set(websocketpp_URL2  "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/websocketpp-b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip")
  set(websocketpp_HASH "SHA256=1385135ede8191a7fbef9ec8099e3c5a673d48df0c143958216cd1690567f583")

  # If you don't have access to the Internet,
  # please pre-download websocketpp
  set(possible_file_locations
    $ENV{HOME}/Downloads/websocketpp-b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip
    ${CMAKE_SOURCE_DIR}/websocketpp-b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip
    ${CMAKE_BINARY_DIR}/websocketpp-b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip
    /tmp/websocketpp-b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip
    /star-fj/fangjun/download/github/websocketpp-b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(websocketpp_URL  "${f}")
      file(TO_CMAKE_PATH "${websocketpp_URL}" websocketpp_URL)
      message(STATUS "Found local downloaded websocketpp: ${websocketpp_URL}")
      set(websocketpp_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(websocketpp
    URL
      ${websocketpp_URL}
      ${websocketpp_URL2}
    URL_HASH          ${websocketpp_HASH}
  )

  FetchContent_GetProperties(websocketpp)
  if(NOT websocketpp_POPULATED)
    message(STATUS "Downloading websocketpp from ${websocketpp_URL}")
    FetchContent_Populate(websocketpp)
  endif()
  message(STATUS "websocketpp is downloaded to ${websocketpp_SOURCE_DIR}")
  # add_subdirectory(${websocketpp_SOURCE_DIR} ${websocketpp_BINARY_DIR} EXCLUDE_FROM_ALL)
  include_directories(${websocketpp_SOURCE_DIR})
endfunction()

download_websocketpp()
