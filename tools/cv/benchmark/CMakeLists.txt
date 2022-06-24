# using opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(${CMAKE_CURRENT_LIST_DIR}/)

add_executable(opencv_bench ${CMAKE_CURRENT_LIST_DIR}/opencv_benchmark.cpp)
target_link_libraries(opencv_bench MNNOpenCV ${OpenCV_LIBS})
