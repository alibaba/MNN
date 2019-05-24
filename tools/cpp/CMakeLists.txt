# put output to build dir
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)

add_executable(MNNV2Basic.out MNNV2Basic.cpp revertMNNModel.cpp)
target_link_libraries(MNNV2Basic.out ${MNN_DEPEND})

include_directories(../../3rd_party/imageHelper/)
add_executable(mobilenetTest.out mobilenetTest.cpp )
target_link_libraries(mobilenetTest.out ${MNN_DEPEND})

add_executable(backendTest.out backendTest.cpp)
target_link_libraries(backendTest.out ${MNN_DEPEND})

add_executable(testModel.out testModel.cpp)
target_link_libraries(testModel.out ${MNN_DEPEND})

add_executable(testModelWithDescrisbe.out testModelWithDescrisbe.cpp Config.cpp)
target_link_libraries(testModelWithDescrisbe.out ${MNN_DEPEND})

add_executable(getPerformance.out getPerformance.cpp)
target_link_libraries(getPerformance.out ${MNN_DEPEND})

# include converter's MNN_generated.h
include_directories(..)
add_executable(timeProfile.out timeProfile.cpp revertMNNModel.cpp Profiler.hpp Profiler.cpp)
target_link_libraries(timeProfile.out ${MNN_DEPEND})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_executable(checkDir.out checkDir.cpp)
    add_executable(checkFile.out checkFile.cpp)
    add_executable(winogradExample.out winogradExample.cpp)
    target_link_libraries(winogradExample.out ${MNN_DEPEND})
    add_executable(winogradGenerateGLSL.out winogradGenerateGLSL.cpp)
    target_link_libraries(winogradGenerateGLSL.out ${MNN_DEPEND})
    add_executable(winogradGenerateCL.out winogradGenerateCL.cpp)
    target_link_libraries(winogradGenerateCL.out ${MNN_DEPEND})	
endif()
