include_directories(${CMAKE_SOURCE_DIR})
add_library(sherpa-mnn-c-api c-api.cc)
target_link_libraries(sherpa-mnn-c-api sherpa-mnn-core)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(sherpa-mnn-c-api PUBLIC SHERPA_MNN_BUILD_SHARED_LIBS=1)
  target_compile_definitions(sherpa-mnn-c-api PUBLIC SHERPA_MNN_BUILD_MAIN_LIB=1)
endif()

add_library(sherpa-mnn-cxx-api cxx-api.cc)
target_link_libraries(sherpa-mnn-cxx-api sherpa-mnn-c-api)

install(
  TARGETS
    sherpa-mnn-c-api
    sherpa-mnn-cxx-api
  DESTINATION
    lib
)

install(
  FILES
    c-api.h
    cxx-api.h
  DESTINATION
    include/sherpa-mnn/c-api
)
