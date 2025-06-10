set(ESPEAK_CONFIG_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(ESPEAK_CONFIG_H ${ESPEAK_CONFIG_DIR}/config.h)
configure_file(config.h.in ${ESPEAK_CONFIG_H})

add_library(espeak-ng-config INTERFACE)
target_include_directories(espeak-ng-config INTERFACE ${ESPEAK_CONFIG_DIR})

add_library(espeak-ng
  common.c
  mnemonics.c
  error.c
  ieee80.c

  compiledict.c

  dictionary.c
  encoding.c
  intonation.c
  langopts.c
  numbers.c
  phoneme.c
  phonemelist.c
  readclause.c
  setlengths.c
  soundicon.c
  ssml.c
  synthdata.c
  synthesize.c
  tr_languages.c
  translate.c
  translateword.c
  voices.c
  wavegen.c
  speech.c

  espeak_api.c
)
set_target_properties(
  espeak-ng PROPERTIES
  MACOSX_RPATH ON
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

target_include_directories(espeak-ng BEFORE PRIVATE $<TARGET_PROPERTY:espeak-include,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(espeak-ng BEFORE PRIVATE $<TARGET_PROPERTY:espeak-include2,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(espeak-ng PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include/compat)

if (NOT MSVC)
  target_compile_options(espeak-ng PRIVATE
    "-fPIC"
    "-fvisibility=hidden"
    "-fno-exceptions"
    "-fwrapv"

    "-pedantic"

    "-Wunused-parameter"
    "-Wunused"
    "-Wuninitialized"
    "-Wreturn-type"
    "-Wmissing-prototypes"
    "-Wint-conversion"
    "-Wimplicit"
    "-Wmisleading-indentation"
  )
endif()
target_compile_definitions(espeak-ng PRIVATE "LIBESPEAK_NG_EXPORT=1")
if (NOT BUILD_SHARED_LIBS)
  target_compile_definitions(espeak-ng INTERFACE "LIBESPEAK_NG_EXPORT=1")
endif()

target_compile_definitions(espeak-ng PRIVATE "PATH_ESPEAK_DATA=\"${CMAKE_INSTALL_PREFIX}/share/espeak-ng-data\"")

if (USE_ASYNC)
  target_sources(espeak-ng PRIVATE
    event.c
    fifo.c
    espeak_command.c
  )
endif(USE_ASYNC)

if (USE_MBROLA)
  target_sources(espeak-ng PRIVATE
    mbrowrap.c
    compilembrola.c
    synth_mbrola.c
  )
endif(USE_MBROLA)

if (USE_KLATT)
  target_sources(espeak-ng PRIVATE klatt.c)
endif(USE_KLATT)

# if (USE_SPEECHPLAYER)
#   target_sources(espeak-ng PRIVATE sPlayer.c)
#   target_link_libraries(espeak-ng PRIVATE speechPlayer)
# endif(USE_SPEECHPLAYER)

if (HAVE_LIBSONIC AND USE_LIBSONIC)
  target_link_libraries(espeak-ng PRIVATE ${SONIC_LIB})
  target_include_directories(espeak-ng PRIVATE ${SONIC_INC})
endif()

if (HAVE_LIBPCAUDIO AND USE_LIBPCAUDIO)
  target_link_libraries(espeak-ng PRIVATE ${PCAUDIO_LIB})
  target_include_directories(espeak-ng PRIVATE ${PCAUDIO_INC})
  if (APPLE)
    target_link_libraries(espeak-ng PRIVATE "-framework CoreAudio" "-framework AudioToolbox" "-framework AudioUnit")
  endif()
endif()

target_link_libraries(espeak-ng PRIVATE espeak-ng-config ucd)
if (NOT MSVC)
  target_link_libraries(espeak-ng PRIVATE m)
endif()
target_link_libraries(espeak-ng PUBLIC espeak-include espeak-include2)

if ((MINGW OR WIN32 OR MSVC) AND BUILD_SHARED_LIBS)
  set_target_properties(espeak-ng PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ..
  )
endif()

if (MINGW)
  target_link_options(espeak-ng PRIVATE "-static-libstdc++" "-static")
endif()

set_target_properties(espeak-ng PROPERTIES
  SOVERSION ${PROJECT_VERSION_MAJOR}
  VERSION ${PROJECT_VERSION}
)

install(TARGETS espeak-ng LIBRARY)
