IF(MNN_EVALUATION)
  add_executable(classficationTopkEval.out ${CMAKE_CURRENT_LIST_DIR}/classficationTopkEval.cpp)
  target_link_libraries(classficationTopkEval.out ${MNN_DEPS})
ENDIF()
