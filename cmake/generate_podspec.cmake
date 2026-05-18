# =============================================================================
# generate_podspec.cmake
# 从 CMake target 中自动提取源文件列表，生成 MNN.podspec
#
# 使用方法:
#   在 CMakeLists.txt 末尾（所有 target 定义之后）:
#   include(cmake/generate_podspec.cmake)
#   generate_mnn_podspec_from_targets()
# =============================================================================

cmake_minimum_required(VERSION 3.6)

# =============================================================================
# Configuration
# =============================================================================
# Use MNN_VERSION from CMakeLists.txt as default
if(NOT DEFINED MNN_PODSPEC_VERSION OR MNN_PODSPEC_VERSION STREQUAL "")
    set(MNN_PODSPEC_VERSION "${MNN_VERSION}" CACHE STRING "MNN podspec version" FORCE)
endif()
set(MNN_IOS_DEPLOYMENT_TARGET "11.0" CACHE STRING "iOS deployment target")

# =============================================================================
# 辅助函数：获取 target 的所有源文件
# =============================================================================
function(get_target_sources_recursive TARGET OUTPUT_VAR)
    set(RESULT "")

    # 获取 target 的源文件
    get_target_property(SRCS ${TARGET} SOURCES)
    if(SRCS)
        list(APPEND RESULT ${SRCS})
    endif()

    # 获取 OBJECT 库的源文件
    get_target_property(INTERFACE_SOURCES ${TARGET} INTERFACE_SOURCES)
    if(INTERFACE_SOURCES)
        list(APPEND RESULT ${INTERFACE_SOURCES})
    endif()

    set(${OUTPUT_VAR} ${RESULT} PARENT_SCOPE)
endfunction()

# =============================================================================
# Helper: Convert absolute paths to relative paths (filter out directories)
# =============================================================================
function(convert_to_relative_paths SOURCE_LIST ROOT_DIR OUTPUT_VAR)
    set(RESULT "")
    foreach(SRC ${SOURCE_LIST})
        if(IS_ABSOLUTE ${SRC})
            # Skip directories
            if(IS_DIRECTORY ${SRC})
                continue()
            endif()
            file(RELATIVE_PATH REL_PATH "${ROOT_DIR}" "${SRC}")
            list(APPEND RESULT "${REL_PATH}")
        else()
            # Skip if it looks like a directory (no extension)
            get_filename_component(EXT "${SRC}" EXT)
            if(NOT EXT STREQUAL "")
                list(APPEND RESULT "${SRC}")
            endif()
        endif()
    endforeach()
    set(${OUTPUT_VAR} ${RESULT} PARENT_SCOPE)
endfunction()

# =============================================================================
# 辅助函数：按扩展名分类文件
# =============================================================================
function(classify_files FILE_LIST CPP_OUT HEADER_OUT ASM_OUT MM_OUT METAL_OUT OTHER_OUT)
    set(CPP_FILES "")
    set(HEADER_FILES "")
    set(ASM_FILES "")
    set(MM_FILES "")
    set(METAL_FILES "")
    set(OTHER_FILES "")

    foreach(FILE ${FILE_LIST})
        get_filename_component(EXT "${FILE}" EXT)
        string(TOLOWER "${EXT}" EXT_LOWER)

        if(EXT_LOWER MATCHES "\\.(cpp|cc|c)$")
            list(APPEND CPP_FILES "${FILE}")
        elseif(EXT_LOWER MATCHES "\\.(h|hpp)$")
            list(APPEND HEADER_FILES "${FILE}")
        elseif(EXT_LOWER MATCHES "\\.(s|S)$")
            list(APPEND ASM_FILES "${FILE}")
        elseif(EXT_LOWER STREQUAL ".mm")
            list(APPEND MM_FILES "${FILE}")
        elseif(EXT_LOWER STREQUAL ".metal")
            list(APPEND METAL_FILES "${FILE}")
        else()
            list(APPEND OTHER_FILES "${FILE}")
        endif()
    endforeach()

    set(${CPP_OUT} ${CPP_FILES} PARENT_SCOPE)
    set(${HEADER_OUT} ${HEADER_FILES} PARENT_SCOPE)
    set(${ASM_OUT} ${ASM_FILES} PARENT_SCOPE)
    set(${MM_OUT} ${MM_FILES} PARENT_SCOPE)
    set(${METAL_OUT} ${METAL_FILES} PARENT_SCOPE)
    set(${OTHER_OUT} ${OTHER_FILES} PARENT_SCOPE)
endfunction()

# =============================================================================
# 辅助函数：格式化文件列表为 podspec 格式（每行一个）
# =============================================================================
function(format_file_list_for_podspec FILE_LIST OUTPUT_VAR INDENT)
    set(RESULT "")
    list(LENGTH FILE_LIST LEN)
    math(EXPR LAST_INDEX "${LEN} - 1")
    set(INDEX 0)

    foreach(FILE ${FILE_LIST})
        if(INDEX EQUAL LAST_INDEX)
            string(APPEND RESULT "${INDENT}'${FILE}'")
        else()
            string(APPEND RESULT "${INDENT}'${FILE}',\n")
        endif()
        math(EXPR INDEX "${INDEX} + 1")
    endforeach()

    set(${OUTPUT_VAR} "${RESULT}" PARENT_SCOPE)
endfunction()

# =============================================================================
# 验证函数：检查生成 podspec 的前置条件
# =============================================================================
function(validate_podspec_generation)
    set(MNN_ROOT "${CMAKE_SOURCE_DIR}")

    # 1. 检查 MNN_ROOT 目录是否存在
    if(NOT IS_DIRECTORY "${MNN_ROOT}")
        message(FATAL_ERROR "MNN source directory not found: ${MNN_ROOT}")
    endif()

    # 2. 检查必要目录是否存在
    set(REQUIRED_DIRS "include/MNN" "source" "3rd_party")
    foreach(dir ${REQUIRED_DIRS})
        if(NOT IS_DIRECTORY "${MNN_ROOT}/${dir}")
            message(FATAL_ERROR "Required directory not found: ${dir}\nPlease ensure MNN source structure is complete.")
        endif()
    endforeach()

    # 3. 检查 LICENSE 文件是否存在
    if(NOT EXISTS "${MNN_ROOT}/LICENSE")
        message(WARNING "LICENSE file not found. Podspec license field may be invalid.")
    endif()

    # 4. 检查 MNN_TARGETS 是否已定义且非空
    if(NOT DEFINED MNN_TARGETS)
        message(FATAL_ERROR "MNN_TARGETS is not defined. Please include generate_podspec.cmake after all targets are created.")
    endif()
    list(LENGTH MNN_TARGETS TARGETS_COUNT)
    if(TARGETS_COUNT EQUAL 0)
        message(FATAL_ERROR "MNN_TARGETS is empty. No targets to generate podspec from.")
    endif()

    # 5. 检查版本号是否有效
    if(NOT DEFINED MNN_PODSPEC_VERSION OR MNN_PODSPEC_VERSION STREQUAL "")
        message(FATAL_ERROR "MNN_PODSPEC_VERSION is not set. Please define MNN_VERSION or pass -DMNN_PODSPEC_VERSION=x.x.x")
    endif()

    message(STATUS "Validation passed: All prerequisites met for podspec generation.")
endfunction()

# =============================================================================
# 主函数：从 CMake targets 生成 podspec
# =============================================================================
function(generate_mnn_podspec_from_targets)
    set(MNN_ROOT "${CMAKE_SOURCE_DIR}")

    message(STATUS "========================================")
    message(STATUS "Generating MNN.podspec from CMake targets...")
    message(STATUS "========================================")

    # =============================================================================
    # 0. 验证前置条件
    # =============================================================================
    validate_podspec_generation()

    # =============================================================================
    # 1. Collect source files from all MNN targets
    # Using MNN_TARGETS variable defined in CMakeLists.txt (fully automatic!)
    # =============================================================================
    set(ALL_SOURCE_FILES "")
    set(ARM82_SOURCE_FILES "")

    # MNN_TARGETS is already defined in CMakeLists.txt and contains all targets
    # that are linked to the final MNN library. We just iterate over them!
    message(STATUS "Collecting sources from MNN_TARGETS: ${MNN_TARGETS}")

    foreach(TGT ${MNN_TARGETS})
        if(TARGET ${TGT})
            get_target_property(SRCS ${TGT} SOURCES)
            if(SRCS)
                # ARM82 needs special compiler flags, collect separately
                if("${TGT}" STREQUAL "MNN_Arm82")
                    list(APPEND ARM82_SOURCE_FILES ${SRCS})
                    message(STATUS "  [ARM82] ${TGT}: collecting separately")
                else()
                    list(APPEND ALL_SOURCE_FILES ${SRCS})
                    list(LENGTH SRCS SRC_COUNT)
                    message(STATUS "  ${TGT}: ${SRC_COUNT} files")
                endif()
            endif()
        endif()
    endforeach()

    # Also collect ARM targets (MNNARM64, MNNARM32) which may not be in MNN_TARGETS
    foreach(TGT MNNARM64 MNNARM32)
        if(TARGET ${TGT})
            get_target_property(SRCS ${TGT} SOURCES)
            if(SRCS)
                list(APPEND ALL_SOURCE_FILES ${SRCS})
                list(LENGTH SRCS SRC_COUNT)
                message(STATUS "  ${TGT}: ${SRC_COUNT} files")
            endif()
        endif()
    endforeach()

    # Collect ARM82 if it exists but wasn't in MNN_TARGETS
    if(TARGET MNN_Arm82)
        get_target_property(SRCS MNN_Arm82 SOURCES)
        if(SRCS AND NOT ARM82_SOURCE_FILES)
            list(APPEND ARM82_SOURCE_FILES ${SRCS})
            list(LENGTH SRCS SRC_COUNT)
            message(STATUS "  [ARM82] MNN_Arm82: ${SRC_COUNT} files")
        endif()
    endif()

    # 去重
    list(REMOVE_DUPLICATES ALL_SOURCE_FILES)
    list(REMOVE_DUPLICATES ARM82_SOURCE_FILES)

    # 转换为相对路径
    convert_to_relative_paths("${ALL_SOURCE_FILES}" "${MNN_ROOT}" ALL_SOURCE_FILES_REL)
    convert_to_relative_paths("${ARM82_SOURCE_FILES}" "${MNN_ROOT}" ARM82_SOURCE_FILES_REL)

    # 分类文件
    classify_files("${ALL_SOURCE_FILES_REL}"
        CPP_FILES HEADER_FILES ASM_FILES MM_FILES METAL_FILES OTHER_FILES)
    classify_files("${ARM82_SOURCE_FILES_REL}"
        ARM82_CPP ARM82_HEADER ARM82_ASM ARM82_MM ARM82_METAL ARM82_OTHER)

    # 统计
    list(LENGTH ALL_SOURCE_FILES_REL TOTAL_FILES)
    list(LENGTH ARM82_SOURCE_FILES_REL ARM82_TOTAL_FILES)
    message(STATUS "Collected ${TOTAL_FILES} source files from CMake targets")
    message(STATUS "Collected ${ARM82_TOTAL_FILES} ARM82 files (separate subspec)")



    # =============================================================================
    # 3. 收集公开头文件
    # =============================================================================
    file(GLOB_RECURSE PUBLIC_HEADERS
        "${MNN_ROOT}/include/MNN/*.h"
        "${MNN_ROOT}/include/MNN/*.hpp"
    )
    convert_to_relative_paths("${PUBLIC_HEADERS}" "${MNN_ROOT}" PUBLIC_HEADERS_REL)

    # =============================================================================
    # 4. 格式化文件列表
    # =============================================================================
    format_file_list_for_podspec("${ALL_SOURCE_FILES_REL}" SOURCE_FILES_FORMATTED "    ")
    format_file_list_for_podspec("${PUBLIC_HEADERS_REL}" PUBLIC_HEADERS_FORMATTED "    ")
    format_file_list_for_podspec("${ARM82_SOURCE_FILES_REL}" ARM82_FILES_FORMATTED "      ")

    # 收集 MM 文件路径（用于 requires_arc）
    set(ARC_FILES "")
    foreach(FILE ${MM_FILES})
        list(APPEND ARC_FILES "${FILE}")
    endforeach()
    format_file_list_for_podspec("${ARC_FILES}" ARC_FILES_FORMATTED "    ")

    # =============================================================================
    # 5. 生成 podspec 内容
    # =============================================================================
    set(PODSPEC_CONTENT "# =============================================================================
# MNN.podspec
# Auto-generated by CMake, DO NOT EDIT!
# Command: cmake .. -DMNN_GENERATE_PODSPEC=ON
# =============================================================================

Pod::Spec.new do |s|
  s.name         = \"MNN\"
  s.version      = \"${MNN_PODSPEC_VERSION}\"
  s.summary      = \"MNN - A lightweight deep neural network inference framework\"

  s.description  = <<-DESC
                   MNN is a blazing fast, lightweight deep neural network inference engine.
                   Auto-generated from CMake configuration.
                   DESC

  s.homepage     = \"https://github.com/alibaba/MNN\"
  s.license      = { :type => 'Apache License, Version 2.0', :file => 'LICENSE' }
  s.author       = { \"MNN\" => \"mnn@alibaba-inc.com\" }

  s.platform     = :ios
  s.ios.deployment_target = '${MNN_IOS_DEPLOYMENT_TARGET}'

  # .mm files require ARC
  s.requires_arc = [
${ARC_FILES_FORMATTED}
  ]

  s.source       = { :git => \"https://github.com/alibaba/MNN.git\", :tag => \"\#{s.version}\" }

  s.frameworks   = 'Metal', 'Accelerate', 'CoreVideo', 'Foundation'
  s.weak_frameworks = 'MetalPerformanceShaders'
  s.libraries    = 'c++'

  # =============================================================================
  # Source files - Auto-extracted from CMake targets
  # Total ${TOTAL_FILES} files
  # =============================================================================
  s.source_files = [
    # Public headers
    'include/MNN/*.{h,hpp}',
    'include/MNN/expr/*.{h,hpp}',
    'schema/current/*.{h}',
    '3rd_party/flatbuffers/include/flatbuffers/*.{h}',
    '3rd_party/half/*.{hpp}',

    # Source files extracted from CMake targets
${SOURCE_FILES_FORMATTED}
  ]

  # =============================================================================
  # Public headers
  # =============================================================================
  s.public_header_files = [
${PUBLIC_HEADERS_FORMATTED}
  ]

  # =============================================================================
  # Build configuration
  # =============================================================================
  s.pod_target_xcconfig = {
    'HEADER_SEARCH_PATHS' => [
      '\"\$(PODS_TARGET_SRCROOT)/include\"',
      '\"\$(PODS_TARGET_SRCROOT)/source\"',
      '\"\$(PODS_TARGET_SRCROOT)/express\"',
      '\"\$(PODS_TARGET_SRCROOT)/tools\"',
      '\"\$(PODS_TARGET_SRCROOT)/schema/current\"',
      '\"\$(PODS_TARGET_SRCROOT)/3rd_party\"',
      '\"\$(PODS_TARGET_SRCROOT)/3rd_party/flatbuffers/include\"',
      '\"\$(PODS_TARGET_SRCROOT)/3rd_party/half\"',
      '\"\$(PODS_TARGET_SRCROOT)/source/backend/cpu\"',
      '\"\$(PODS_TARGET_SRCROOT)/source/backend/cpu/arm\"',
      '\"\$(PODS_TARGET_SRCROOT)/source/backend/arm82\"',
      '\"\$(PODS_TARGET_SRCROOT)/source/backend/arm82/compute\"',
      '\"\$(PODS_TARGET_SRCROOT)/source/backend/arm82/asm\"',
      '\"\$(PODS_TARGET_SRCROOT)/source/backend/metal\"',
      '\"\$(PODS_TARGET_SRCROOT)/tools/cv/include\"',
      '\"\$(PODS_TARGET_SRCROOT)/transformers/llm/engine/include\"',
      '\"\$(PODS_TARGET_SRCROOT)/transformers/llm/engine/src\"'
    ].join(' '),
    'GCC_PREPROCESSOR_DEFINITIONS' => [
      '\$(inherited)',
      'MNN_METAL_ENABLED=1',
      'ENABLE_ARMV82=1',
      'MNN_USE_NEON=1',
      'MNN_SUPPORT_TRANSFORMER_FUSE=1',
      'MNN_LOW_MEMORY=1',
      'MNN_IMGPROC_DRAW=1',
      'MNN_REDUCE_SIZE=1'
    ].join(' '),
    'OTHER_CFLAGS' => '-fno-rtti -fno-exceptions -Oz',
    'OTHER_CPLUSPLUSFLAGS' => '-std=c++17 -fno-rtti -fno-exceptions -Oz',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'METAL_LIBRARY_FILE_BASE' => 'mnn',
    'ARCHS' => 'arm64',
    'VALID_ARCHS' => 'arm64',
    # Optimization: remove debug info and strip
    'GCC_OPTIMIZATION_LEVEL' => 'z',
    'DEBUG_INFORMATION_FORMAT' => 'dwarf',
    'STRIP_INSTALLED_PRODUCT' => 'YES',
    'STRIP_STYLE' => 'non-global',
    'DEAD_CODE_STRIPPING' => 'YES',
    'DEPLOYMENT_POSTPROCESSING' => 'YES'
  }

  # =============================================================================
  # ARM82 FP16 - Requires -march=armv8.2-a+fp16 compiler flag
  # Total ${ARM82_TOTAL_FILES} files
  # =============================================================================
  s.subspec 'ARM82' do |sp|
    sp.source_files = [
${ARM82_FILES_FORMATTED}
    ]
    sp.pod_target_xcconfig = {
      'OTHER_CFLAGS' => '-march=armv8.2-a+fp16 -fno-rtti -fno-exceptions',
      'OTHER_CPLUSPLUSFLAGS' => '-std=c++17 -march=armv8.2-a+fp16 -fno-rtti -fno-exceptions'
    }
    sp.xcconfig = {
      'OTHER_ASFLAGS' => '-march=armv8.2-a+fp16'
    }
  end

  s.default_subspecs = ['ARM82']

end
")

    # =============================================================================
    # 6. 写入文件
    # =============================================================================
    set(PODSPEC_FILE "${MNN_ROOT}/MNN.podspec")
    file(WRITE "${PODSPEC_FILE}" "${PODSPEC_CONTENT}")

    message(STATUS "========================================")
    message(STATUS "Generated: ${PODSPEC_FILE}")
    message(STATUS "Total files: ${TOTAL_FILES}")
    message(STATUS "ARM82 files: ${ARM82_TOTAL_FILES}")
    message(STATUS "========================================")
endfunction()

# =============================================================================
# 如果直接调用
# =============================================================================
if(MNN_GENERATE_PODSPEC)
    generate_mnn_podspec_from_targets()
endif()