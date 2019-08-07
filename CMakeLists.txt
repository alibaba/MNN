cmake_minimum_required(VERSION 2.8)

project(MNN)

# complier options
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
enable_language(ASM)
# set(CMAKE_C_COMPILER gcc)
# set(CMAKE_CXX_COMPILER g++)

option(MNN_USE_CPP11 "Enable MNN use c++11" ON)

if (NOT MSVC)
    if(MNN_USE_CPP11)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif()
endif()

# build options
option(MNN_BUILD_HARD "Build -mfloat-abi=hard or not" OFF)
option(MNN_BUILD_SHARED_LIBS "MNN build shared or static lib" ON)
option(MNN_FORBID_MULTI_THREAD "Disable Multi Thread" OFF)
option(MNN_OPENMP "Enable Multiple Thread Linux|Android" ON)
option(MNN_USE_THREAD_POOL "Use Multiple Thread by Self ThreadPool" ON)
option(MNN_SUPPORT_TRAIN "Enable Train Ops" OFF)
option(MNN_BUILD_DEMO "Build demo/exec or not" OFF)
option(MNN_BUILD_QUANTOOLS "Build Quantized Tools or not" OFF)
option(MNN_EVALUATION "Build Evaluation Tools or not" OFF)

if (MNN_USE_THREAD_POOL)
    set(MNN_OPENMP OFF)
    add_definitions(-DMNN_USE_THREAD_POOL)
endif()

if(MNN_FORBID_MULTI_THREAD)
    add_definitions(-DMNN_FORBIT_MULTI_THREADS)
endif()

# debug options
option(MNN_DEBUG "Enable MNN DEBUG" OFF)
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(MNN_DEBUG ON)
endif()
option(MNN_DEBUG_MEMORY "MNN Debug Memory Access" OFF)
option(MNN_DEBUG_TENSOR_SIZE "Enable Tensor Size" OFF )
option(MNN_GPU_TRACE "Enable MNN Gpu Debug" OFF)

if(MNN_DEBUG_MEMORY)
    add_definitions(-DMNN_DEBUG_MEMORY)
endif()
if(MNN_DEBUG_TENSOR_SIZE)
    add_definitions(-DMNN_DEBUG_TENSOR_SIZE)
endif()
if(MNN_GPU_TRACE)
    add_definitions(-DMNN_GPU_FORCE_FINISH)
endif()

# backend options
option(MNN_METAL "Enable Metal" OFF)
option(MNN_OPENCL "Enable OpenCL" OFF)
option(MNN_OPENGL "Enable OpenGL" OFF)
option(MNN_VULKAN "Enable Vulkan" OFF)
option(MNN_ARM82 "Enable ARM82" OFF)

# codegen register ops
if (MNN_METAL)
    add_definitions(-DMNN_CODEGEN_REGISTER)
endif()

# target options
option(MNN_BUILD_BENCHMARK "Build benchmark or not" ON)
option(MNN_BUILD_TEST "Build tests or not" OFF)
option(MNN_BUILD_FOR_ANDROID_COMMAND "Build from command" OFF)
option(MNN_BUILD_FOR_IOS "Build for ios" OFF)
set (MNN_HIDDEN FALSE)
if (NOT MNN_BUILD_TEST)
    if (NOT MNN_DEBUG)
        set (MNN_HIDDEN TRUE)
    endif()
endif()

include(cmake/macros.cmake)

message(STATUS ">>>>>>>>>>>>>")
message(STATUS "MNN BUILD INFO:")
message(STATUS "\tSystem: ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tDEBUG: ${MNN_DEBUG}")
message(STATUS "\tMetal: ${MNN_METAL}")
message(STATUS "\tOpenCL: ${MNN_OPENCL}")
message(STATUS "\tOpenGL: ${MNN_OPENGL}")
message(STATUS "\tVulkan: ${MNN_VULKAN}")
message(STATUS "\tOpenMP: ${MNN_OPENMP}")
message(STATUS "\tHideen: ${MNN_HIDDEN}")

# flags
if(SYSTEM.Android)
    add_definitions(-DMNN_BUILD_FOR_ANDROID)
    if(PROCESSOR.arm)
        add_definitions(-mfloat-abi=softfp -mfpu=neon)
    endif()
endif()

if(SYSTEM.Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
        add_definitions(-mfpu=neon)	#please define in project/cross-compile/arm.toolchain.cmake
    endif()
    if(MNN_BUILD_HARD)
        add_definitions(-mfloat-abi=hard)  #better define in project/cross-compile/arm.toolchain.cmake
    endif()
endif()

if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /MT")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /MT")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MT")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MTd")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /MT")
    set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} /MT")
	set(CMAKE_CXX_STANDARD 14)
elseif(SYSTEM.Android OR SYSTEM.Linux)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
    add_definitions(-fPIC)
elseif()
endif()

if(MNN_DEBUG)
    add_definitions(-DMNN_DEBUG)
    if(MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEBUG /DEBUG")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG /DEBUG")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEBUG -g")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG -g")
    endif()
else()
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
        if(MNN_BUILD_FOR_ANDROID_COMMAND)
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s -Wl,--gc-sections")
        endif()
    endif()
endif()
if ((MNN_HIDDEN) AND (NOT MSVC))
    add_definitions(-fvisibility=hidden)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
endif()

# include files
set(MNN.Path source)
set(MNN.Source_DIR
    ${MNN.Path}
    ${MNN.Path}/core
    ${MNN.Path}/backend/cpu
    ${MNN.Path}/backend/cpu/arm
    ${MNN.Path}/backend/cpu/compute
    ${MNN.Path}/cv
    ${MNN.Path}/math
    ${MNN.Path}/shape
)
include_directories(
    "include/" 
    "schema/current" 
    "3rd_party/flatbuffers/include"
)

if(MNN_METAL)
    set (MNN.Source_DIR ${MNN.Source_DIR} ${MNN.Path}/backend/metal)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "^Linux")
    if(PROCESSOR.arm OR PROCESSOR.aarch64)
        set(aarch64_linux_include
            #/usr/include/c++/4.9
            #/usr/lib/gcc/x86_64-linux-gnu/4.9
            #/usr/lib/gcc/x86_64-linux-gnu/4.9/include
            #/usr/include/x86_64-linux-gnu/c++/4.9
        )
        include_directories(${aarch64_linux_include})
    endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x64)|(X64)|(amd64)|(AMD64)")
    add_definitions(-DMNN_USE_SSE)
    set (MNN.Source_DIR ${MNN.Source_DIR} ${MNN.Path}/backend/cpu/sse)
endif()

# *.c
foreach(file_path ${MNN.Source_DIR})
    file(GLOB file_source_c ${file_path}/*.c ${file_path}/*.h)
    set(MNN.Source_C ${MNN.Source_C} ${file_source_c})
endforeach()
message(STATUS "[*] searching *.c")

# *.cc
foreach(file_path ${MNN.Source_DIR})
    file(GLOB file_source_cc ${file_path}/*.cc ${file_path}/*.hpp)
    set(MNN.Source_CC ${MNN.Source_CC} ${file_source_cc})
endforeach()
message(STATUS "[*] searching *.cc")

# *.cpp
foreach(file_path ${MNN.Source_DIR})
    file(GLOB file_source_cpp ${file_path}/*.cpp)
    set(MNN.Source_CPP ${MNN.Source_CPP} ${file_source_cpp})
endforeach()
message(STATUS "[*] searching *.cpp")

if(MNN_METAL)
    # *.m *.mm
    foreach(file_path ${MNN.Source_DIR})
        file(GLOB file_source_m ${file_path}/*.mm ${file_path}/*.m ${file_path}/*.metal)
        set(MNN.Source_M ${MNN.Source_M} ${file_source_m})
    endforeach()
    message(STATUS "[*] searching *.m *.mm")
endif()

set(GLOBAL.SOURCE ${GLOBAL.SOURCE} ${MNN.Source_C} ${MNN.Source_CPP} ${MNN.Source_M} ${MNN.Source_CC})
set(GLOBAL.SOURCE_INCLUDE ${GLOBAL.SOURCE_INCLUDE} ${MNN.Source_DIR})

# *.s *.S assembly
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
    set(MNN.Source_Assembly_DIR ${MNN.Path}/backend/cpu/arm/arm32)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64")
    set(MNN.Source_Assembly_DIR ${MNN.Path}/backend/cpu/arm/arm64)
elseif(ARCH_NAME STREQUAL "ios_fat_arm")
    set(MNN.Source_Assembly_DIR ${MNN.Source_Assembly_DIR} ${MNN.Path}/backend/cpu/arm/arm64)
    set(MNN.Source_Assembly_DIR ${MNN.Source_Assembly_DIR} ${MNN.Path}/backend/cpu/arm/arm32)
endif()
message(STATUS "[*] searching *.s *.S")

foreach(file_path ${MNN.Source_Assembly_DIR})
    file(GLOB file_source_assembly ${file_path}/*.s ${file_path}/*.S)
    set(MNN.Source_Assembly ${MNN.Source_Assembly} ${file_source_assembly})
endforeach()

# set assembly file use c compiler flags
set_property(SOURCE ${MNN.Source_Assembly} PROPERTY LANGUAGE C)

include_directories(${MNN.Source_DIR})
include_directories("./")

if(SYSTEM.Android AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NATIVE_LIBRARY_OUTPUT}/${ANDROID_ABI})
    add_definitions(-DMNN_USE_LOGCAT)
endif()

if(MNN_BUILD_FOR_IOS)
    add_library(MNN ${GLOBAL.SOURCE} ${MNN.Source_Assembly} ${Metal.Resource_Metal})
    set_target_properties(MNN PROPERTIES FRAMEWORK TRUE)
    set_target_properties(MNN PROPERTIES
        MACOSX_FRAMEWORK_IDENTIFIER com.alibaba.MNN
        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
        )
    set_target_properties(MNN PROPERTIES MACOSX_FRAMEWORK_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/project/ios/MNN/Info.plist)
    # set_source_files_properties($Metal.Resource_Metal} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
    # set_target_properties(MNN PROPERTIES RESOURCE ${Metal.Resource_Metal})
else()
    if(MNN_BUILD_SHARED_LIBS)
        add_library(MNN SHARED ${GLOBAL.SOURCE} ${MNN.Source_Assembly})
        if (WIN32)
            install(TARGETS MNN RUNTIME DESTINATION lib)
        else()
            install(TARGETS MNN LIBRARY DESTINATION lib)
        endif()
    else()
        add_library(MNN STATIC ${GLOBAL.SOURCE} ${MNN.Source_Assembly})
        install(TARGETS MNN ARCHIVE DESTINATION lib)
    endif()
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -fPIE -s")
endif()


# pthread lib
if(SYSTEM.Linux)
    target_link_libraries(MNN pthread)
endif()

set(MNN_DEPEND MNN)
if(NOT MNN_BUILD_SHARED_LIBS)
    if (BUILD_IOS OR APPLE)
        set(MNN_DEPEND -all_load, ${MNN_DEPEND}, -noall_load)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(MNN_DEPEND -Wl,--whole-archive ${MNN_DEPEND} -Wl,--no-whole-archive)
    endif()
endif()

if(MNN_VULKAN)
    set(MNN_DEPEND ${MNN_DEPEND} MNN_Vulkan)
    add_subdirectory(${MNN.Path}/backend/vulkan)
    message(STATUS "[*] linking MNN with Vulkan done")
endif()
if(MNN_OPENGL)
    set(MNN_DEPEND ${MNN_DEPEND} MNN_GL)
    add_subdirectory(${MNN.Path}/backend/opengl)
    message(STATUS "[*] linking MNN with OpenGL done")
endif()
if(MNN_OPENCL)
    set(MNN_DEPEND ${MNN_DEPEND} MNN_CL)
    add_subdirectory(${MNN.Path}/backend/opencl)
    message(STATUS "[*] linking MNN with OpenCL done")
endif()
if (MNN_ARM82)
    set(MNN_DEPEND ${MNN_DEPEND} MNN_Arm82)
    add_subdirectory(${MNN.Path}/backend/arm82)
    message(STATUS "[*] linking MNN with ARM 82 done")
endif()

if (BUILD_IOS OR APPLE)
else()
    if(MNN_OPENMP)
        message(STATUS "[*] Checking OpenMP")
        find_package(OpenMP)
        # For CMake < 3.9, we need to make the target ourselves
        if(NOT TARGET OpenMP::OpenMP_CXX)
            find_package(Threads REQUIRED)
            add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
            set_property(TARGET OpenMP::OpenMP_CXX
                PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
            # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
            set_property(TARGET OpenMP::OpenMP_CXX
                PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
        endif()
        if (WIN32)
            set(OpenMP_C_FLAGS "/openmp ${OpenMP_C_FLAGS}")
            set(OpenMP_CXX_FLAGS "/openmp ${OpenMP_CXX_FLAGS}")
        endif()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

# Android thrid party
if(CMAKE_SYSTEM_NAME MATCHES "^Android")
    target_link_libraries(MNN log)
endif()

# Linux library
if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
    set (CMAKE_FIND_ROOT_PATH C:/MinGW)
endif()

# Darwin third party
if(MNN_BUILD_FOR_IOS OR MNN_METAL)
    target_link_libraries(MNN "-framework Foundation")
    target_link_libraries(MNN "-framework Metal")
endif()

if(WIN32)
    target_compile_definitions(MNN PRIVATE "-DBUILDING_DLL")
    target_compile_definitions(MNN PUBLIC "-D_CRT_SECURE_NO_WARNINGS")
    target_compile_options(MNN PUBLIC "/wd4244" "/wd4146" "/wd4018" "/wd4267" "/wd4996" "/wd4081")
endif()

if(SYSTEM.Android AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    set(MNN_INCLUDE_OUTPUT ${NATIVE_INCLUDE_OUTPUT}/MNN)
    add_custom_command(
        TARGET MNN
        POST_BUILD
        COMMAND mkdir -p ${MNN_INCLUDE_OUTPUT}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_SRC}/include/MNNDefine.h ${MNN_INCLUDE_OUTPUT}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_SRC}/include/Interpreter.hpp ${MNN_INCLUDE_OUTPUT}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_SRC}/include/MNNForwardType.h ${MNN_INCLUDE_OUTPUT}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_SRC}/include/HalideRuntime.h ${MNN_INCLUDE_OUTPUT}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_SRC}/include/Tensor.hpp ${MNN_INCLUDE_OUTPUT}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_SRC}/include/AutoTime.hpp ${MNN_INCLUDE_OUTPUT}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_SRC}/include/ErrorCode.hpp ${MNN_INCLUDE_OUTPUT}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_SRC}/include/ImageProcess.hpp ${MNN_INCLUDE_OUTPUT}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_SRC}/include/Matrix.h ${MNN_INCLUDE_OUTPUT}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${ROOT_SRC}/include/Rect.h ${MNN_INCLUDE_OUTPUT}

    )
else()
    if (MNN_BUILD_DEMO)
        add_subdirectory(demo/exec)
    endif()
    add_subdirectory(tools/cpp)
endif()

if (MNN_SUPPORT_TRAIN)
    add_subdirectory(tools/train)
endif()

if(MNN_BUILD_TEST)
    add_subdirectory(test)
endif()

if(MNN_BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()
if(MNN_BUILD_QUANTOOLS)
    add_subdirectory(tools/quantization)
endif()

if(MNN_EVALUATION)
    add_subdirectory(tools/evaluation)
endif()

# schema generator
if(WIN32)
    add_custom_target( MNN_SCHEMA ALL
        COMMAND powershell ${CMAKE_CURRENT_SOURCE_DIR}/schema/generate.ps1 -lazy
    )
else()
    add_custom_target( MNN_SCHEMA ALL
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/schema/generate.sh -lazy
    )
endif()
add_dependencies(MNN MNN_SCHEMA)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/MNNDefine.h   DESTINATION include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/Interpreter.hpp DESTINATION include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/HalideRuntime.h DESTINATION include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/Tensor.hpp      DESTINATION include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/ErrorCode.hpp   DESTINATION include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/ImageProcess.hpp DESTINATION include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/Matrix.h      DESTINATION include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/Rect.h   DESTINATION include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/MNNForwardType.h   DESTINATION include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/AutoTime.hpp   DESTINATION include)

