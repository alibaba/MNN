# put output to build dir
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)

add_executable(MNNV2Basic.out MNNV2Basic.cpp revertMNNModel.cpp)
target_link_libraries(MNNV2Basic.out ${MNN_DEPEND})
install(TARGETS MNNV2Basic.out RUNTIME DESTINATION demo)
install(FILES MNNV2Basic.cpp DESTINATION demo)

add_executable(backendTest.out backendTest.cpp)
target_link_libraries(backendTest.out ${MNN_DEPEND})

option(ENABLE_PERFORMANCE "Enable Performance" OFF)
if(ENABLE_PERFORMANCE)
    add_definitions(-DMNN_USE_OPENCL_WRAPPER)
    SET(OPENCL_PATH ../../Source/backend/opencl)
    include_directories(
        ${OPENCL_PATH}
        ${OPENCL_PATH}/3rdParty
        ${OPENCL_PATH}/3rdParty/Half
    )
    file(GLOB_RECURSE OPENCL_SRCS ${OPENCL_PATH}/*.cpp ${OPENCL_PATH}/*.cc)
    add_executable(getPerformance.out getPerformance.cpp ${OPENCL_SRCS})
    target_link_libraries(getPerformance.out ${MNN_DEPEND})
endif()

add_executable(testModel.out testModel.cpp)
target_link_libraries(testModel.out ${MNN_DEPEND})
install(TARGETS testModel.out RUNTIME DESTINATION demo)
install(FILES testModel.cpp DESTINATION demo)

add_executable(testModelWithDescrisbe.out testModelWithDescrisbe.cpp Config.cpp)
target_link_libraries(testModelWithDescrisbe.out ${MNN_DEPEND})

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
