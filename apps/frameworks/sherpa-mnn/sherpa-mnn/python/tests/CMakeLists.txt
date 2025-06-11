function(sherpa_onnx_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "${name}_py")

  add_test(NAME ${name}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
    WORKING_DIRECTORY
      ${CMAKE_CURRENT_SOURCE_DIR}
  )

  get_filename_component(sherpa_onnx_path ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  set_property(TEST ${name}
    PROPERTY ENVIRONMENT "PYTHONPATH=${sherpa_onnx_path}:$<TARGET_FILE_DIR:_sherpa_mnn>:$ENV{PYTHONPATH}"
  )
endfunction()

# please sort the files in alphabetic order
set(py_test_files
  test_fast_clustering.py
  test_feature_extractor_config.py
  test_keyword_spotter.py
  test_offline_recognizer.py
  test_online_recognizer.py
  test_online_transducer_model_config.py
  test_speaker_recognition.py
  test_text2token.py
)

foreach(source IN LISTS py_test_files)
  sherpa_onnx_add_py_test(${source})
endforeach()

