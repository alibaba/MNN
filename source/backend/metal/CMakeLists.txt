FILE(GLOB MNN_Metal_SRC ${CMAKE_CURRENT_LIST_DIR}/*.mm ${CMAKE_CURRENT_LIST_DIR}/*.hpp ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
# The decoder is only reachable through the packed-shader accessors, so keep it
# out of builds that embed the shader text directly.
list(REMOVE_ITEM MNN_Metal_SRC ${CMAKE_CURRENT_LIST_DIR}/MetalShaderCodec.cpp)
IF(MNN_SUPPORT_RENDER)
    file(GLOB MNN_Metal_Render_SRC ${CMAKE_CURRENT_LIST_DIR}/render/*.mm ${CMAKE_CURRENT_LIST_DIR}/render/*.hpp ${CMAKE_CURRENT_LIST_DIR}/render/*.cpp)
    list(APPEND MNN_Metal_SRC ${MNN_Metal_Render_SRC})
ENDIF()
FILE(GLOB MNN_Metal_KERNELS_SRC ${CMAKE_CURRENT_LIST_DIR}/*.metal)

IF(MNN_METAL_PACK_SHADER)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(MNN_METAL_PACK_DIR "${CMAKE_CURRENT_BINARY_DIR}/packed_shader")
    set(MNN_METAL_PACK_CPP "${MNN_METAL_PACK_DIR}/MetalPackedShader.cpp")
    set(MNN_METAL_PACK_HPP "${MNN_METAL_PACK_DIR}/MetalPackedShader.hpp")
    # The shader text lives in these headers, so regenerate whenever they change.
    set(MNN_METAL_PACK_INPUTS
        ${CMAKE_CURRENT_LIST_DIR}/ConvSimdGroupShader.hpp
        ${CMAKE_CURRENT_LIST_DIR}/MetalAttentionShader.hpp
        ${CMAKE_CURRENT_LIST_DIR}/MetalLinearAttentionShader.hpp
        ${CMAKE_CURRENT_LIST_DIR}/LayerNormSimdGroupShader.hpp
        ${CMAKE_CURRENT_LIST_DIR}/MetalFlashAttnShader.hpp
        ${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp)
    add_custom_command(
        OUTPUT ${MNN_METAL_PACK_CPP} ${MNN_METAL_PACK_HPP}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/packshader.py
                ${CMAKE_CURRENT_LIST_DIR} ${MNN_METAL_PACK_CPP} ${MNN_METAL_PACK_HPP}
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/packshader.py ${MNN_METAL_PACK_INPUTS}
        COMMENT "Packing Metal shader source"
        VERBATIM)
    list(APPEND MNN_Metal_SRC ${MNN_METAL_PACK_CPP} ${MNN_METAL_PACK_HPP}
        ${CMAKE_CURRENT_LIST_DIR}/MetalShaderCodec.cpp)
ENDIF()

add_library(MNNMetal OBJECT ${MNN_Metal_SRC} "${CMAKE_CURRENT_LIST_DIR}/MetalOPRegister.mm")
target_compile_options(MNNMetal PRIVATE -DMNN_METAL_ENABLED=1)
IF(MNN_METAL_PACK_SHADER)
  target_compile_options(MNNMetal PRIVATE -DMNN_METAL_PACK_SHADER)
  target_include_directories(MNNMetal PRIVATE ${MNN_METAL_PACK_DIR} ${CMAKE_CURRENT_LIST_DIR})
ENDIF()
if(MNN_METAL_TENSOR)
  target_compile_options(MNNMetal PRIVATE -DMNN_METAL_TENSOR)
endif()
if(MNN_METAL_OP_PROFILE)
  target_compile_options(MNNMetal PRIVATE -DMNN_METAL_OP_PROFILE=1)
endif()
set_property(TARGET MNNMetal APPEND_STRING PROPERTY COMPILE_FLAGS "-fobjc-arc")

list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNMetal>)
