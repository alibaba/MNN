message(STATUS "MNN Depends:" ${MNN_DEPS})
set(MNN_EXEC_DEMOS "")

add_executable(pictureRecognition.out ${CMAKE_CURRENT_LIST_DIR}/pictureRecognition.cpp)
list(APPEND MNN_EXEC_DEMOS pictureRecognition.out)
add_executable(pictureRecognition_module.out ${CMAKE_CURRENT_LIST_DIR}/pictureRecognition_module.cpp)
list(APPEND MNN_EXEC_DEMOS pictureRecognition_module.out)
add_executable(multithread_imgrecog.out ${CMAKE_CURRENT_LIST_DIR}/multithread_imgrecog.cpp)
list(APPEND MNN_EXEC_DEMOS multithread_imgrecog.out)
add_executable(pictureRecognition_batch.out ${CMAKE_CURRENT_LIST_DIR}/pictureRecognition_batch.cpp)
list(APPEND MNN_EXEC_DEMOS pictureRecognition_batch.out)

add_executable(pictureRotate.out ${CMAKE_CURRENT_LIST_DIR}/pictureRotate.cpp)
list(APPEND MNN_EXEC_DEMOS pictureRotate.out)

add_executable(multiPose.out ${CMAKE_CURRENT_LIST_DIR}/multiPose.cpp)
list(APPEND MNN_EXEC_DEMOS multiPose.out)

add_executable(segment.out ${CMAKE_CURRENT_LIST_DIR}/segment.cpp)
list(APPEND MNN_EXEC_DEMOS segment.out)

add_executable(expressDemo.out ${CMAKE_CURRENT_LIST_DIR}/expressDemo.cpp)
list(APPEND MNN_EXEC_DEMOS expressDemo.out)

add_executable(expressMakeModel.out ${CMAKE_CURRENT_LIST_DIR}/expressMakeModel.cpp)
list(APPEND MNN_EXEC_DEMOS expressMakeModel.out)

add_executable(transformerDemo.out ${CMAKE_CURRENT_LIST_DIR}/transformerDemo.cpp)
list(APPEND MNN_EXEC_DEMOS transformerDemo.out)

add_executable(nluDemo.out ${CMAKE_CURRENT_LIST_DIR}/nluDemo.cpp)
list(APPEND MNN_EXEC_DEMOS nluDemo.out)

foreach(TARGET ${MNN_EXEC_DEMOS})
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
