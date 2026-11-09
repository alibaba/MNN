add_executable(api_test
  cplusplus.cpp
  harness.c
  harness.h
  main.c
)
include_directories(
  ${PROJECT_SOURCE_DIR}/src/include
  ${PROJECT_BINARY_DIR}/src
  ${PROJECT_SOURCE_DIR}/extensions/include
  ${PROJECT_BINARY_DIR}/extensions
)
target_link_libraries(api_test PRIVATE
  libcmark-gfm
  libcmark-gfm-extensions)

add_test(NAME api_test COMMAND api_test)
if(WIN32 AND BUILD_SHARED_LIBS)
  set_tests_properties(api_test PROPERTIES
    ENVIRONMENT "PATH=$<TARGET_FILE_DIR:libcmark-gfm>;$<TARGET_FILE_DIR:libcmark-gfm-extensions>;$ENV{PATH}")
endif()
