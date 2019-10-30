# put output to build dir
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)
include_directories(../../3rd_party/imageHelper/)
add_executable(pictureRecognition.out pictureRecognition.cpp)
target_link_libraries(pictureRecognition.out ${MNN_DEPEND})
add_executable(pictureRotate.out pictureRotate.cpp)
target_link_libraries(pictureRotate.out MNN)

add_executable(multiPose.out multiPose.cpp)
target_link_libraries(multiPose.out ${MNN_DEPEND})

add_executable(segment.out segment.cpp)
target_link_libraries(segment.out MNN_Express ${MNN_DEPEND})

add_executable(expressDemo.out expressDemo.cpp)
target_link_libraries(expressDemo.out MNN_Express ${MNN_DEPEND})
