set(MNN_CPP_TOOLS "")
add_executable(GetMNNInfo ${CMAKE_CURRENT_LIST_DIR}/GetMNNInfo.cpp)
list(APPEND MNN_CPP_TOOLS GetMNNInfo)
add_executable(ModuleBasic.out ${CMAKE_CURRENT_LIST_DIR}/ModuleBasic.cpp)
list(APPEND MNN_CPP_TOOLS ModuleBasic.out)
IF(MNN_OPENGL AND MNN_OPENCL)
    add_definitions(-DMNN_USE_LIB_WRAPPER)
    add_library(MNN_CL_WRAP OBJECT ${CMAKE_CURRENT_LIST_DIR}/../../source/backend/opencl/core/runtime/OpenCLWrapper.cpp)
    add_executable(GpuInterTest.out ${CMAKE_CURRENT_LIST_DIR}/GpuInterTest.cpp )
    target_link_libraries(GpuInterTest.out MNN_CL_WRAP)
    list(APPEND MNN_CPP_TOOLS GpuInterTest.out)
ENDIF()
add_executable(SequenceModuleTest.out ${CMAKE_CURRENT_LIST_DIR}/SequenceModuleTest.cpp)
list(APPEND MNN_CPP_TOOLS SequenceModuleTest.out)

add_executable(mergeInplaceForCPU ${CMAKE_CURRENT_LIST_DIR}/mergeInplaceForCPU.cpp)
list(APPEND MNN_CPP_TOOLS mergeInplaceForCPU)

add_executable(MNNV2Basic.out ${CMAKE_CURRENT_LIST_DIR}/MNNV2Basic.cpp)
list(APPEND MNN_CPP_TOOLS MNNV2Basic.out)
if (MNN_USE_SSE)
    target_compile_options(MNNV2Basic.out PRIVATE -DMNN_USE_SSE)
endif()

add_executable(mobilenetTest.out ${CMAKE_CURRENT_LIST_DIR}/mobilenetTest.cpp )
list(APPEND MNN_CPP_TOOLS mobilenetTest.out)

add_executable(backendTest.out ${CMAKE_CURRENT_LIST_DIR}/backendTest.cpp)
list(APPEND MNN_CPP_TOOLS backendTest.out)

add_executable(testModel.out ${CMAKE_CURRENT_LIST_DIR}/testModel.cpp)
list(APPEND MNN_CPP_TOOLS testModel.out)

add_executable(testModel_expr.out ${CMAKE_CURRENT_LIST_DIR}/testModel_expr.cpp)
list(APPEND MNN_CPP_TOOLS testModel_expr.out)

add_executable(testModelWithDescribe.out ${CMAKE_CURRENT_LIST_DIR}/testModelWithDescribe.cpp)
list(APPEND MNN_CPP_TOOLS testModelWithDescribe.out)

add_executable(getPerformance.out ${CMAKE_CURRENT_LIST_DIR}/getPerformance.cpp)
list(APPEND MNN_CPP_TOOLS getPerformance.out)

add_executable(checkInvalidValue.out ${CMAKE_CURRENT_LIST_DIR}/checkInvalidValue.cpp)
list(APPEND MNN_CPP_TOOLS checkInvalidValue.out)

add_executable(timeProfile.out ${CMAKE_CURRENT_LIST_DIR}/timeProfile.cpp ${CMAKE_CURRENT_LIST_DIR}/revertMNNModel.cpp ${CMAKE_CURRENT_LIST_DIR}/Profiler.cpp)
list(APPEND MNN_CPP_TOOLS timeProfile.out)

add_executable(testTrain.out ${CMAKE_CURRENT_LIST_DIR}/testTrain.cpp)
list(APPEND MNN_CPP_TOOLS testTrain.out)

add_executable(fuseTest ${CMAKE_CURRENT_LIST_DIR}/fuseTest.cpp)
list(APPEND MNN_CPP_TOOLS fuseTest)

add_executable(LoRA ${CMAKE_CURRENT_LIST_DIR}/LoRA.cpp)
list(APPEND MNN_CPP_TOOLS LoRA)

foreach(TARGET ${MNN_CPP_TOOLS})
    target_link_libraries(${TARGET} ${MNN_DEPS})
    if (MSVC)
        target_compile_definitions(${TARGET} PRIVATE "_CRT_SECURE_NO_WARNINGS")
        if (NOT MNN_BUILD_SHARED_LIBS)
            foreach (DEPEND ${MNN_DEPS})
                target_link_options(${TARGET} PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
            endforeach ()
        endif()
    endif()
endforeach()

if (NOT WIN32)
    add_executable(checkDir.out ${CMAKE_CURRENT_LIST_DIR}/checkDir.cpp)
    add_executable(checkFile.out ${CMAKE_CURRENT_LIST_DIR}/checkFile.cpp)
    add_executable(winogradExample.out ${CMAKE_CURRENT_LIST_DIR}/winogradExample.cpp)
    target_link_libraries(winogradExample.out ${MNN_DEPS})
endif()
