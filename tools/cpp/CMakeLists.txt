add_executable(MNNV2Basic.out ${CMAKE_CURRENT_LIST_DIR}/MNNV2Basic.cpp ${CMAKE_CURRENT_LIST_DIR}/revertMNNModel.cpp)
target_link_libraries(MNNV2Basic.out ${MNN_DEPS})

add_executable(mobilenetTest.out ${CMAKE_CURRENT_LIST_DIR}/mobilenetTest.cpp )
target_link_libraries(mobilenetTest.out ${MNN_DEPS})

add_executable(backendTest.out ${CMAKE_CURRENT_LIST_DIR}/backendTest.cpp)
target_link_libraries(backendTest.out ${MNN_DEPS})

add_executable(testModel.out ${CMAKE_CURRENT_LIST_DIR}/testModel.cpp)
target_link_libraries(testModel.out ${MNN_DEPS})

add_executable(testModelWithDescrisbe.out ${CMAKE_CURRENT_LIST_DIR}/testModelWithDescrisbe.cpp ${CMAKE_CURRENT_LIST_DIR}/Config.cpp)
target_link_libraries(testModelWithDescrisbe.out ${MNN_DEPS})

add_executable(getPerformance.out ${CMAKE_CURRENT_LIST_DIR}/getPerformance.cpp)
target_link_libraries(getPerformance.out ${MNN_DEPS})

add_executable(checkInvalidValue.out ${CMAKE_CURRENT_LIST_DIR}/checkInvalidValue.cpp)
target_link_libraries(checkInvalidValue.out ${MNN_DEPS})

add_executable(timeProfile.out ${CMAKE_CURRENT_LIST_DIR}/timeProfile.cpp ${CMAKE_CURRENT_LIST_DIR}/revertMNNModel.cpp ${CMAKE_CURRENT_LIST_DIR}/Profiler.cpp)
target_link_libraries(timeProfile.out ${MNN_DEPS})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_executable(checkDir.out ${CMAKE_CURRENT_LIST_DIR}/checkDir.cpp)
    add_executable(checkFile.out ${CMAKE_CURRENT_LIST_DIR}/checkFile.cpp)
    add_executable(winogradExample.out ${CMAKE_CURRENT_LIST_DIR}/winogradExample.cpp)
    target_link_libraries(winogradExample.out ${MNN_DEPS})
    add_executable(winogradGenerateGLSL.out ${CMAKE_CURRENT_LIST_DIR}/winogradGenerateGLSL.cpp)
    target_link_libraries(winogradGenerateGLSL.out ${MNN_DEPS})
    add_executable(winogradGenerateCL.out ${CMAKE_CURRENT_LIST_DIR}/winogradGenerateCL.cpp)
    target_link_libraries(winogradGenerateCL.out ${MNN_DEPS})
endif()
