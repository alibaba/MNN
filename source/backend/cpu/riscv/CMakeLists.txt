if(NOT DEFINED ARCHS)
    set(ARCHS ${CMAKE_SYSTEM_PROCESSOR})
endif()

file(GLOB MNN_RVV_SRC
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rvv/*.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64" OR ARCHS STREQUAL "riscv64")
    if(MNN_USE_RVV)
        include(CheckCXXSourceCompiles)

        set(MNN_RVV_TEST_REQUIRED_FLAGS_BACKUP "${CMAKE_REQUIRED_FLAGS}")
        set(MNN_RVV_TRY_COMPILE_TARGET_TYPE_BACKUP "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

        set(CMAKE_REQUIRED_FLAGS
            "${MNN_RVV_TEST_REQUIRED_FLAGS_BACKUP} -march=rv64gcv -mabi=lp64d"
        )

        check_cxx_source_compiles("
        #include <riscv_vector.h>
        vfloat32m8_t test(vfloat32m8_t s, vfloat32m8_t w, float v, size_t vl) {
            return __riscv_vfmacc_vf_f32m8(s, v, w, vl);
        }
        int main() { return 0; }" MNN_RVV_AVAILABLE)

        set(CMAKE_REQUIRED_FLAGS
            "${MNN_RVV_TEST_REQUIRED_FLAGS_BACKUP} -march=rv64gcv_zfh_zvfh -mabi=lp64d -D__fp16=_Float16"
        )

        check_cxx_source_compiles("
        #include <riscv_vector.h>
        vfloat16m8_t test(vfloat16m8_t s, vfloat16m8_t w, __fp16 v, size_t vl) {
            return __riscv_vfmacc_vf_f16m8(s, v, w, vl);
        }
        int main() { return 0; }" MNN_RVV_ZVFH_AVAILABLE)

        set(CMAKE_REQUIRED_FLAGS "${MNN_RVV_TEST_REQUIRED_FLAGS_BACKUP}")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE "${MNN_RVV_TRY_COMPILE_TARGET_TYPE_BACKUP}")

        if(MNN_RVV_AVAILABLE)
            if(MNN_RVV_ZVFH_AVAILABLE)
                set(MNN_RVV_ARCH_FLAGS
                    -march=rv64gcv_zfh_zvfh
                    -mabi=lp64d
                )
                message(STATUS "Enabling RVV/Zvfh optimizations")
            else()
                set(MNN_RVV_ARCH_FLAGS
                    -march=rv64gcv
                    -mabi=lp64d
                )
                message(STATUS "Enabling RVV optimizations without Zvfh")
            endif()

            add_compile_definitions(MNN_USE_RVV)
            add_library(MNNRVV OBJECT ${MNN_RVV_SRC})
            target_include_directories(MNNRVV PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}
                ${CMAKE_CURRENT_LIST_DIR}/rvv/
            )
            target_compile_options(MNNRVV PRIVATE ${MNN_RVV_ARCH_FLAGS})

            list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNRVV>)
            list(APPEND MNN_TARGETS MNNRVV)
        else()
            message(WARNING "MNN_USE_RVV is ON, but the toolchain does not support RVV intrinsics")
        endif()
    endif()
endif()