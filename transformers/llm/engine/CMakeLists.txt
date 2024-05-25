# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

if (MSVC)
    # compile static lib, surrpot Winwows
    add_library(llm STATIC ${SRCS})
    target_link_libraries(llm ${MNN_DEPS})
else()
    # compile dynamic so, support Linux/Mac
    add_library(llm SHARED ${SRCS})
    set_target_properties(llm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    target_link_libraries(llm ${MNN_DEPS})
endif()
target_compile_features(llm PRIVATE cxx_std_17)

add_executable(llm_demo ${CMAKE_CURRENT_LIST_DIR}/llm_demo.cpp)
target_compile_features(llm_demo PRIVATE cxx_std_17)
target_link_libraries(llm_demo llm)