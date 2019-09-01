if(SYSTEM.Android AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NATIVE_LIBRARY_OUTPUT}/${ANDROID_ABI})
endif()

file(GLOB_RECURSE SRCS *.cc *.hpp *.h *.cpp)

if (MNN_BUILD_SHARED_LIBS AND (NOT MSVC) AND (NOT WIN32))
    add_library(MNN_CL SHARED ${SRCS})
    target_link_libraries(MNN_CL PRIVATE MNN)
else()
    add_library(MNN_CL STATIC ${SRCS})
endif()

if((NOT MSVC) AND (NOT WIN32))
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions")
    target_compile_options(MNN_CL PRIVATE -Wno-deprecated-declarations -Wno-ignored-attributes)
else()
    target_compile_options(MNN_CL PRIVATE "/wd4267" "/wd4018" "/wd4251" "/wd4996" "/wd4244")
endif()

target_include_directories(MNN_CL PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Android")
    add_definitions(-DMNN_USE_OPENCL_WRAPPER)
    #add_definitions(-DENABLE_OPENCL_TURNING_PROFILER)
    #add_definitions(-DLOG_VERBOSE)
else()
    find_package(OpenCL)
    target_link_libraries(MNN_CL PRIVATE ${OpenCL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()
