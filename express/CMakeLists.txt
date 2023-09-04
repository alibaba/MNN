

set (EXTRA_LIBS "")
if(MNN_CUDA_PROFILE)
    set(CUDA_MIN_VERSION "9.0")
    find_package(CUDA ${CUDA_MIN_VERSION})

    MESSAGE("cuda dir is:${CUDA_INCLUDE_DIRS}")
    include_directories(
        ${CUDA_INCLUDE_DIRS}
    )
     set(EXTRA_LIBS ${CUDA_INCLUDE_DIRS}/../lib/libnvToolsExt.so)
endif()

file(GLOB_RECURSE MNN_EXPR_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.*")
option(MNN_EXPR_SHAPE_EAGER "Force compute Expr's shape directly cost" OFF)
IF (MNN_EXPR_SHAPE_EAGER)
    add_definitions(-DMNN_EXPR_SHAPE_EAGER)
ENDIF()
IF(MNN_SEP_BUILD)
    if (MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../")
    endif()
    if (MNN_BUILD_MINI)
        add_library(MNN_Express SHARED ${MNN_EXPR_SRCS} $<TARGET_OBJECTS:MNNTransform>)
        target_compile_options(MNN_Express PRIVATE -DMNN_BUILD_MINI)
    else()
        add_library(MNN_Express SHARED ${MNN_EXPR_SRCS})
    endif()
    target_link_libraries(MNN_Express MNN ${EXTRA_LIBS})
    install(TARGETS MNN_Express
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
ELSE()
    add_library(MNN_Express OBJECT ${MNN_EXPR_SRCS})
ENDIF()


