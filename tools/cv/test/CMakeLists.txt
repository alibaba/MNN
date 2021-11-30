# using opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

# using gtest
INCLUDE(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# copy test images
FILE(COPY imgs DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
FILE(COPY imgs DESTINATION ${CMAKE_BINARY_DIR}/)

include_directories(${CMAKE_CURRENT_LIST_DIR}/)

enable_testing()

file(GLOB_RECURSE IMGPROC_SRC ${CMAKE_CURRENT_LIST_DIR}/imgproc/*.cpp)
if(MNN_IMGCODECS)
    file(GLOB_RECURSE IMGCODECS_SRC ${CMAKE_CURRENT_LIST_DIR}/imgcodecs/*.cpp)
endif()
add_executable(opencv_test ${IMGPROC_SRC} ${IMGCODECS_SRC} ./main.cpp)
target_link_libraries(opencv_test MNNOpenCV gtest_main ${OpenCV_LIBS})

include(GoogleTest)
gtest_discover_tests(opencv_test)
