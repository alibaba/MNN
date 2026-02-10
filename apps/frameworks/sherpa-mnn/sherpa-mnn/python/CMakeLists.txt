add_subdirectory(csrc)

if(SHERPA_MNN_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
