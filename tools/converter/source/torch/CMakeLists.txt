file(GLOB TORCH_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
add_library(MNNConverterTorch OBJECT ${TORCH_SRC})

IF (CMAKE_SYSTEM_NAME MATCHES "Linux" AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(LIB_TORCH_PATH "${CMAKE_CURRENT_BINARY_DIR}/libtorch/share/cmake")
    IF (EXISTS "/etc/redhat-release")
        # redhat force _GLIBCXX_USE_CXX11_ABI=0
        SET(LIB_TORCH_NAME "libtorch-shared-with-deps-1.9.0%2Bcpu.zip")
    ELSE()
        # ubuntu MNN use _GLIBCXX_USE_CXX11_ABI=1
        SET(LIB_TORCH_NAME "libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip")
    ENDIF()
    STRING(REPLACE "%2B" "+" LIB_TORCH_ZIP ${LIB_TORCH_NAME})
    IF (NOT EXISTS ${LIB_TORCH_PATH})
        IF (NOT EXISTS ${LIB_TORCH_ZIP})
            execute_process(COMMAND wget "https://download.pytorch.org/libtorch/cpu/${LIB_TORCH_NAME}")
        ENDIF()
        execute_process(COMMAND unzip ${LIB_TORCH_ZIP} -d ${CMAKE_CURRENT_BINARY_DIR}
                        OUTPUT_VARIABLE TORCH_PATH
                        RESULT_VARIABLE EXIT_CODE
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
    ENDIF()
ELSE()
    execute_process(COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path)"
                    OUTPUT_VARIABLE LIB_TORCH_PATH
                    RESULT_VARIABLE EXIT_CODE
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT ${EXIT_CODE} EQUAL 0)
        message(FATAL_ERROR
                "The \"torch\" Python3 package is not installed. Please install it using the following command: \"pip3 install torch\".")
    endif()
ENDIF()

message(STATUS "LibTorch Path is : " ${LIB_TORCH_PATH})
find_package(Torch REQUIRED PATHS ${LIB_TORCH_PATH} NO_DEFAULT_PATH)

target_include_directories(MNNConverterTorch PRIVATE ${TORCH_INCLUDE_DIRS})
set_target_properties(MNNConverterTorch PROPERTIES CXX_STANDARD 14)
target_compile_options(MNNConverterTorch PRIVATE "-fexceptions")
target_compile_options(MNNConverterTorch PRIVATE "-frtti")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET MNNConverterTorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:MNNConverterTorch>)
endif (MSVC)
list(APPEND MNN_CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:MNNConverterTorch>)
