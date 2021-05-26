if(APPLE AND MNN_AAPL_FMWK)
    install(TARGETS MNN
        EXPORT MNNTargets
        CONFIGURATIONS Debug Release RelWithDebInfo
        FRAMEWORK DESTINATION .
        )
    set(FRAMEWORK_NAME "MNN.framework")
    set(TARGET_CONFIG_DIR ./${FRAMEWORK_NAME}/CMake/MNN)
    set(INCLUDE_INSTALL_DIR Headers/)
    set(LIB_INSTALL_DIR .)
elseif(ANDROID)
    install(TARGETS MNN
        EXPORT MNNTargets
        CONFIGURATIONS Debug Release RelWithDebInfo
        RUNTIME DESTINATION lib/${ANDROID_ABI}
        LIBRARY DESTINATION lib/${ANDROID_ABI}
        ARCHIVE DESTINATION lib/${ANDROID_ABI}
        )
    set(TARGET_CONFIG_DIR lib/${ANDROID_ABI}/cmake/MNN)
    set(INCLUDE_INSTALL_DIR include/)
    set(LIB_INSTALL_DIR lib/${ANDROID_ABI})
else()
    install(TARGETS MNN
        EXPORT MNNTargets
        CONFIGURATIONS Debug Release RelWithDebInfo
        RUNTIME DESTINATION lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
    set(TARGET_CONFIG_DIR lib/cmake/MNN)
    set(INCLUDE_INSTALL_DIR include/)
    set(LIB_INSTALL_DIR lib/)
endif()


install(EXPORT MNNTargets
    FILE MNNTargets.cmake
    DESTINATION ${TARGET_CONFIG_DIR}
    )

include(CMakePackageConfigHelpers)


configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MNNConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MNNConfig.cmake
    INSTALL_DESTINATION ${TARGET_CONFIG_DIR}
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
    )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MNNConfigVersion.cmake
    VERSION 1.1.6
    COMPATIBILITY AnyNewerVersion 
    )

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/MNNConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MNNConfigVersion.cmake
    DESTINATION ${TARGET_CONFIG_DIR}
    )
