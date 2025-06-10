cmake_minimum_required(VERSION 3.8)

project(espeak-ng
  VERSION 1.52.0.1
  DESCRIPTION "open source speech synthesizer that supports more than hundred languages and accents"
  HOMEPAGE_URL "https://github.com/espeak-ng/espeak-ng"
)
set(USE_MBROLA OFF)
set(USE_KLATT OFF)
set(USE_SPEECHPLAYER OFF)
set(USE_LIBSONIC OFF)
set(USE_LIBPCAUDIO OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(COMPILE_INTONATIONS "Intonation compilation" OFF)
option(ENABLE_TESTS "Enable tests" OFF)

include(cmake/config.cmake)
add_subdirectory(src)
if (COMPILE_INTONATIONS AND NOT CMAKE_CROSSCOMPILING)
  include(cmake/data.cmake)
endif()
# include(cmake/docs.cmake)

# include(cmake/package.cmake)
# include(CPack)

# if(ENABLE_TESTS AND NOT CMAKE_CROSSCOMPILING)
#   add_subdirectory(tests)
# endif()

message(STATUS "Configuration:")
message(STATUS "  shared: ${BUILD_SHARED_LIBS}")
message(STATUS "  mbrola: ${USE_MBROLA} (${MBROLA_BIN})")
message(STATUS "  libsonic: ${USE_LIBSONIC} (${SONIC_LIB} ${SONIC_INC})")
message(STATUS "  libpcaudio: ${USE_LIBPCAUDIO} (${PCAUDIO_LIB} ${PCAUDIO_INC})")
message(STATUS "  klatt: ${USE_KLATT}")
message(STATUS "  speech-player: ${USE_SPEECHPLAYER}")
message(STATUS "  async: ${USE_ASYNC}")

# install(
#   DIRECTORY vim/ftdetect vim/syntax DESTINATION share/vim/vimfiles
# )