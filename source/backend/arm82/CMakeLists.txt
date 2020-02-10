if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR IOS_ARCH STREQUAL "arm64")
    file(GLOB MNN_ARM82_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
    file(GLOB MNN_ARM82_SRCS_ASM "${CMAKE_CURRENT_LIST_DIR}/asm/arm64/*")
    IF(MNN_SEP_BUILD)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
        add_library(
            MNN_Arm82
            SHARED
            ${MNN_ARM82_SRCS}
            ${MNN_ARM82_SRCS_ASM}
            )
        target_link_libraries(MNN_Arm82 MNN)
    ELSE()
        add_library(
            MNN_Arm82
            OBJECT
            ${MNN_ARM82_SRCS}
            ${MNN_ARM82_SRCS_ASM}
            )
    ENDIF()
    add_definitions(-march=armv8.2a+fp16)
    target_include_directories(MNN_Arm82 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/asm/)
endif()
