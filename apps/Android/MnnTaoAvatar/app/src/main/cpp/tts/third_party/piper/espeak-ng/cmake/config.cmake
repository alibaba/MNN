include(CheckSymbolExists)
check_symbol_exists(mkstemp "stdlib.h" HAVE_MKSTEMP)

option(USE_MBROLA "Use mbrola for speech synthesis" ${HAVE_MBROLA})
option(USE_LIBSONIC "Use libsonit for faster speech rates" ${HAVE_LIBSONIC})
option(USE_LIBPCAUDIO "Use libPcAudio for sound output" ${HAVE_LIBPCAUDIO})

option(USE_KLATT "Use klatt for speech synthesis" ON)
option(USE_SPEECHPLAYER "Use speech-player for speech synthesis" ON)
if (HAVE_PTHREAD)
  option(USE_ASYNC "Support asynchronous speech synthesis" ON)
else()
  set(USE_ASYNC OFF)
endif()

option(ESPEAK_COMPAT "Install compat binary symlinks" OFF)