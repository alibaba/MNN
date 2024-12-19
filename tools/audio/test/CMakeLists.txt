# using gtest
INCLUDE(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# FILE(COPY ../imgs DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

include_directories(${CMAKE_CURRENT_LIST_DIR}/)

enable_testing()

add_executable(audio_test audio_test.cpp)
target_link_libraries(audio_test MNNAudio gtest_main)

include(GoogleTest)
gtest_discover_tests(audio_test)
