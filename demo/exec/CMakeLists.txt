# put output to build dir
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)

if(MNN_FREEIMAGE_VALID)
    include_directories("/usr/local/include")
    link_directories("/usr/local/lib")
    add_executable(pictureRotate.out pictureRotate.cpp)
    add_executable(pictureRecognition.out pictureRecognition.cpp)
    add_executable(multiPose.out multiPose.cpp)
    target_link_libraries(pictureRotate.out freeimage MNN)
    target_link_libraries(pictureRecognition.out freeimage MNN)
    target_link_libraries(multiPose.out freeimage MNN)
endif()

