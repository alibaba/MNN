add_subdirectory(csrc)
if(SHERPA_MNN_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

if(SHERPA_MNN_ENABLE_JNI)
  add_subdirectory(jni)
endif()

if(SHERPA_MNN_ENABLE_C_API)
  add_subdirectory(c-api)
endif()
