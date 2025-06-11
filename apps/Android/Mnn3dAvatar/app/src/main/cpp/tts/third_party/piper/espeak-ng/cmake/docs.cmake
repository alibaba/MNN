find_program(RONN ronn)

option(ESPEAK_BUILD_MANPAGES "Build manpages" ${RONN})

if (RONN AND ESPEAK_BUILD_MANPAGES)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/espeak-ng.1
        COMMAND ${RONN} --roff -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/espeak-ng.1.ronn
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/speak-ng.1
        COMMAND ${RONN} --roff -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/speak-ng.1.ronn
    )

    add_custom_target(
        docs ALL
        DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/espeak-ng.1
            ${CMAKE_CURRENT_BINARY_DIR}/speak-ng.1
    )

    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/espeak-ng.1
            ${CMAKE_CURRENT_BINARY_DIR}/speak-ng.1
        DESTINATION share/man/man1
    )
endif()