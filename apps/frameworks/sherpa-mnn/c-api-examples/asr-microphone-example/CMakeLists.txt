
add_executable(c-api-alsa c-api-alsa.cc alsa.cc)
target_link_libraries(c-api-alsa sherpa-onnx-c-api cargs)

if(DEFINED ENV{SHERPA_MNN_ALSA_LIB_DIR})
  target_link_libraries(c-api-alsa -L$ENV{SHERPA_MNN_ALSA_LIB_DIR} -lasound)
else()
  target_link_libraries(c-api-alsa asound)
endif()
