IF(NOT DEFINED ARCHS)
  set(ARCHS ${CMAKE_SYSTEM_PROCESSOR})
ENDIF()

FILE(GLOB MNN_RVV_SRC ${CMAKE_CURRENT_LIST_DIR}/rvv/*.cpp)

if (MNN_USE_RVV AND (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64" OR ARCHS STREQUAL "riscv64"))
    message(STATUS "MNN_USE_RVV is ON. Checking for RVV support in user-provided compiler flags...")

    set(RVV_CHECK_SOURCE_CONTENT
      "
      #ifndef __riscv_v
      #error \"Compiler does not define __riscv_v. Check your global CMAKE_C(XX)_FLAGS.\"
      #endif
      int main() { return 0; }
      "
    )
    set(TRY_COMPILE_DIR "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeTmp")
    set(RVV_CHECK_FILE_PATH "${TRY_COMPILE_DIR}/rvv_check.cpp")
    file(WRITE "${RVV_CHECK_FILE_PATH}" "${RVV_CHECK_SOURCE_CONTENT}")

    try_compile(
        TOOLCHAIN_SUPPORTS_RVV
        SOURCES "${RVV_CHECK_FILE_PATH}"
    )
    
    file(REMOVE_RECURSE "${TRY_COMPILE_DIR}")

    if(TOOLCHAIN_SUPPORTS_RVV)
        message(STATUS "Toolchain check passed. Enabling RVV source files.")
        add_library(MNNRVV OBJECT ${MNN_RVV_SRC})
        target_include_directories(MNNRVV PRIVATE ${CMAKE_CURRENT_LIST_DIR}/rvv/)
        list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNRVV>)
        list(APPEND MNN_TARGETS MNNRVV)
        add_definitions(-DMNN_USE_RVV)
    else()
        message(FATAL_ERROR "MNN_USE_RVV is ON, but the provided compiler flags do not support RVV. Please provide correct '-march=...v...' and '-mabi=...' flags in CMAKE_C_FLAGS and CMAKE_CXX_FLAGS.")
    endif()
else()
    message(STATUS "RVV optimizations are disabled (MNN_USE_RVV is OFF or target is not riscv64).")
endif()IF(NOT DEFINED ARCHS)
  set(ARCHS ${CMAKE_SYSTEM_PROCESSOR})
ENDIF()

FILE(GLOB MNN_RVV_SRC ${CMAKE_CURRENT_LIST_DIR}/rvv/*.cpp)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64" OR ARCHS STREQUAL "riscv64")
    message(STATUS "Target is riscv64. Performing fine-grained toolchain check for RVV support...")
    set(RVV_TEST_FLAGS "-march=rv64gcv -mabi=lp64d")

    set(RVV_CHECK_SOURCE_CONTENT
      "
      #ifndef __riscv_v
      #error \"Compiler does not define __riscv_v. RVV is not properly supported.\"
      #endif
      int main() { return 0; }
      "
    )
    set(TRY_COMPILE_DIR "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeTmp")
    set(RVV_CHECK_FILE_PATH "${TRY_COMPILE_DIR}/rvv_check.cpp")
    file(WRITE "${RVV_CHECK_FILE_PATH}" "${RVV_CHECK_SOURCE_CONTENT}")

    try_compile(
        TOOLCHAIN_SUPPORTS_RVV
        SOURCES "${RVV_CHECK_FILE_PATH}"
        CMAKE_FLAGS "-DCMAKE_CXX_FLAGS=${RVV_TEST_FLAGS}"
    )

    message(STATUS "Cleaning up temporary compile directory...")
    file(REMOVE_RECURSE "${TRY_COMPILE_DIR}")

    if(TOOLCHAIN_SUPPORTS_RVV)
        message(STATUS "Toolchain check passed. Enabling RVV Optimizations.")
        add_library(MNNRVV OBJECT ${MNN_RVV_SRC})
        target_include_directories(MNNRVV PRIVATE ${CMAKE_CURRENT_LIST_DIR}/rvv/)
        list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNRVV>)
        list(APPEND MNN_TARGETS MNNRVV)
        add_definitions(-DMNN_USE_RVV)
        target_compile_options(MNNRVV PRIVATE ${RVV_TEST_FLAGS})
    else()
        message(WARNING "Target is riscv64, but the toolchain check failed with flags (${RVV_TEST_FLAGS}). Please check if your compiler version supports the 'V' extension. RVV optimizations will be disabled.")
    endif()
else()
    message(STATUS "RVV optimizations are skipped for non-riscv64 targets.")
endif()