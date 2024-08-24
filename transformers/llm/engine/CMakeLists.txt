# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

if (MNN_SEP_BUILD)
    if (MNN_BUILD_SHARED_LIBS)
        # compile dynamic so, support Linux/Mac
        add_library(llm SHARED ${SRCS})
        set_target_properties(llm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        target_link_libraries(llm ${MNN_DEPS})
    else()
        add_library(llm STATIC ${SRCS})
    endif()
else()
    add_library(llm OBJECT ${SRCS})
endif()

add_executable(llm_demo ${CMAKE_CURRENT_LIST_DIR}/llm_demo.cpp)
IF (NOT MNN_SEP_BUILD)
    target_link_libraries(llm_demo ${MNN_DEPS})
ELSE ()
    target_link_libraries(llm_demo ${MNN_DEPS} llm)
ENDIF ()