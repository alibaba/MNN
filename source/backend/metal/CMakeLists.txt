if(MNN_METAL AND APPLE)
    FILE(GLOB MNN_Metal_SRC ${CMAKE_CURRENT_LIST_DIR}/*.mm)
    FILE(GLOB MNN_Metal_KERNELS_SRC ${CMAKE_CURRENT_LIST_DIR}/*.metal)
    message(STATUS "Generating mnn.metallib at ${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib")
    IF(DEFINED SDK_VERSION)
      SET(METAL_SDK_PLAT "iphoneos")
    ELSE()
      SET(METAL_SDK_PLAT "macosx")
    ENDIF()
    message(STATUS "Compiling Metal Kernels with ${METAL_SDK_PLAT} SDK")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib COMMAND xcrun -sdk ${METAL_SDK_PLAT} metal "${MNN_Metal_KERNELS_SRC}" -o ${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib COMMAND_EXPAND_LISTS)
    add_custom_target (MNNMetalLIB DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib COMMENT "Generating mnn.metallib")
    file(REMOVE "${CMAKE_CURRENT_LIST_DIR}/MetalOPRegister.mm")
    add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/MetalOPRegister.mm"
      COMMAND ${PYTHON_EXECUTABLE}
      "${CMAKE_CURRENT_LIST_DIR}/MetalCodeGen.py"
      "${CMAKE_CURRENT_LIST_DIR}/"
      "${CMAKE_CURRENT_LIST_DIR}/MetalOPRegister.mm"
      COMMENT "Metal Code Generation"
    )
    add_library(MNNMetal OBJECT ${MNN_Metal_SRC} "${CMAKE_CURRENT_LIST_DIR}/MetalOPRegister.mm")
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNMetal>)
    list(APPEND MNN_TARGETS MNNMetal)
    add_dependencies(MNNMetal MNNMetalLIB)
    SET(MNN_OBJECTS_TO_LINK "${MNN_OBJECTS_TO_LINK}" PARENT_SCOPE)
    SET(MNN_TARGETS "${MNN_TARGETS}" PARENT_SCOPE)

    # This is just work around some CMake limitations and is really ugly
    #list(APPEND MNN_OBJECTS_TO_LINK ${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib)
    SET(MNN_METALLIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/mnn.metallib" PARENT_SCOPE)
endif()
