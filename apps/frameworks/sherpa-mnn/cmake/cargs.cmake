function(download_cargs)
  include(FetchContent)

  set(cargs_URL "https://github.com/likle/cargs/archive/refs/tags/v1.0.3.tar.gz")
  set(cargs_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/cargs-1.0.3.tar.gz")
  set(cargs_HASH "SHA256=ddba25bd35e9c6c75bc706c126001b8ce8e084d40ef37050e6aa6963e836eb8b")

  # If you don't have access to the Internet,
  # please pre-download cargs
  set(possible_file_locations
    $ENV{HOME}/Downloads/cargs-1.0.3.tar.gz
    ${CMAKE_SOURCE_DIR}/cargs-1.0.3.tar.gz
    ${CMAKE_BINARY_DIR}/cargs-1.0.3.tar.gz
    /tmp/cargs-1.0.3.tar.gz
    /star-fj/fangjun/download/github/cargs-1.0.3.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(cargs_URL  "${f}")
      file(TO_CMAKE_PATH "${cargs_URL}" cargs_URL)
      message(STATUS "Found local downloaded cargs: ${cargs_URL}")
      set(cargs_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(cargs
    URL
      ${cargs_URL}
      ${cargs_URL2}
    URL_HASH
      ${cargs_HASH}
  )

  FetchContent_GetProperties(cargs)
  if(NOT cargs_POPULATED)
    message(STATUS "Downloading cargs ${cargs_URL}")
    FetchContent_Populate(cargs)
  endif()
  message(STATUS "cargs is downloaded to ${cargs_SOURCE_DIR}")
  add_subdirectory(${cargs_SOURCE_DIR} ${cargs_BINARY_DIR} EXCLUDE_FROM_ALL)

  install(TARGETS cargs DESTINATION lib)
  install(FILES ${cargs_SOURCE_DIR}/include/cargs.h
    DESTINATION include
  )
endfunction()

download_cargs()
