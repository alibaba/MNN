FILE(GLOB MNN_Metal_SRC ${CMAKE_CURRENT_LIST_DIR}/*.mm ${CMAKE_CURRENT_LIST_DIR}/*.hpp ${CMAKE_CURRENT_LIST_DIR}/*.h)
FILE(GLOB MNN_Metal_KERNELS_SRC ${CMAKE_CURRENT_LIST_DIR}/*.metal)
message(STATUS "Generating mnn.metallib at ${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib")
IF(DEFINED SDK_VERSION)
    #Defined by iOS toolchain
    SET(METAL_SDK_PLAT "iphoneos")
ELSE()
    SET(METAL_SDK_PLAT "macosx")
ENDIF()
message(STATUS "Compiling Metal Kernels with ${METAL_SDK_PLAT} SDK")
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib COMMAND xcrun -sdk ${METAL_SDK_PLAT} metal "${MNN_Metal_KERNELS_SRC}" -o ${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib COMMAND_EXPAND_LISTS)
add_library(MNNMetal OBJECT ${MNN_Metal_SRC} "${CMAKE_CURRENT_LIST_DIR}/MetalOPRegister.mm")
target_compile_options(MNNMetal PRIVATE -DMNN_METAL_ENABLED=1)
set_property(TARGET MNNMetal APPEND_STRING PROPERTY COMPILE_FLAGS "-fobjc-arc")

list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNMetal>)
# This is just work around some CMake limitations and is really ugly
list(APPEND MNN_OBJECTS_TO_LINK ${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib)
file(REMOVE "${CMAKE_CURRENT_LIST_DIR}/MetalOPRegister.mm")
add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/MetalOPRegister.mm"
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_LIST_DIR}/MetalCodeGen.py"
    "${CMAKE_CURRENT_LIST_DIR}/"
    "${CMAKE_CURRENT_LIST_DIR}/MetalOPRegister.mm"
    COMMENT "Metal Code Generation"
    )
add_custom_target (MNNMetalCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/MetalOPRegister.mm" COMMENT "Registering MetalOps")
add_dependencies(MNNMetal MNNMetalCodeGen)