IF(MNN_BUILD_OPENCV)
  # imgproc submodules start
  option(MNN_IMGPROC_COLOR "Enable imgproc color" ON)
  option(MNN_IMGPROC_GEOMETRIC "Enable imgproc geometric" ON)
  option(MNN_IMGPROC_DRAW "Enable imgproc draw" ON)
  option(MNN_IMGPROC_FILTER "Enable imgproc filter" ON)
  option(MNN_IMGPROC_MISCELLANEOUS "Enable imgproc miscellaneous" ON)
  option(MNN_IMGPROC_STRUCTRAL "Enable imgproc structral" ON)
  option(MNN_IMGPROC_HISTOGRAMS "Enable imgproc histograms" ON)
  # imgproc submodules end
  option(MNN_CVCORE "Enable cv core" ON)
  option(MNN_CALIB3D "Enable calib3d" ON)
  option(MNN_IMGCODECS "Enable imgcodecs" OFF)
  option(MNN_OPENCV_TEST "Enable opencv test" OFF)
  option(MNN_OPENCV_BENCH "Enable opencv benchmark" OFF)

  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/imageHelper/)

  # include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
  if(${MNN_OPENCV_TEST})
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
  endif()

  if(${MNN_OPENCV_BENCH})
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/benchmark)
  endif()

  if(${MNN_OPENCV_TEST} OR ${MNN_OPENCV_BENCH})
    # copy test images
    FILE(COPY imgs DESTINATION ${CMAKE_BINARY_DIR}/)
  endif()

  # file(GLOB_RECURSE IMGPROC_SRC ${CMAKE_CURRENT_LIST_DIR}/source/imgproc/*.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/imgproc/*.hpp)
  set(IMGPROC_SRC "")
  # imgproc color functions
  if(${MNN_IMGPROC_COLOR})
    list(APPEND IMGPROC_SRC ${CMAKE_CURRENT_LIST_DIR}/source/imgproc/color.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/imgproc/color.hpp)
  endif()
  # imgproc geometric functions
  if(${MNN_IMGPROC_GEOMETRIC})
    list(APPEND IMGPROC_SRC ${CMAKE_CURRENT_LIST_DIR}/source/imgproc/geometric.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/imgproc/geometric.hpp)
  endif()
  # imgproc draw functions
  if(${MNN_IMGPROC_DRAW})
    list(APPEND IMGPROC_SRC ${CMAKE_CURRENT_LIST_DIR}/source/imgproc/draw.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/imgproc/draw.hpp)
  endif()
  # imgproc filter functions
  if(${MNN_IMGPROC_FILTER})
    list(APPEND IMGPROC_SRC ${CMAKE_CURRENT_LIST_DIR}/source/imgproc/filter.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/imgproc/filter.hpp)
  endif()
  # imgproc miscellaneous functions
  if(${MNN_IMGPROC_MISCELLANEOUS})
    list(APPEND IMGPROC_SRC ${CMAKE_CURRENT_LIST_DIR}/source/imgproc/miscellaneous.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/imgproc/miscellaneous.hpp)
  endif()
  # imgproc structural functions
  if(${MNN_IMGPROC_STRUCTRAL})
    list(APPEND IMGPROC_SRC ${CMAKE_CURRENT_LIST_DIR}/source/imgproc/structural.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/imgproc/structural.hpp)
  endif()
  # imgproc histograms functions
  if(${MNN_IMGPROC_HISTOGRAMS})
    list(APPEND IMGPROC_SRC ${CMAKE_CURRENT_LIST_DIR}/source/imgproc/histograms.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/imgproc/histograms.hpp)
  endif()
  # calib3d functions
  if(${MNN_CALIB3D})
    add_definitions(-DMNN_CALIB3D)
    file(GLOB_RECURSE CALIB3D_SRC ${CMAKE_CURRENT_LIST_DIR}/source/calib3d/*.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/calib3d.hpp)
  endif()
  # imgcodecs functions
  if(${MNN_IMGCODECS})
    add_definitions(-DMNN_IMGCODECS)
    file(GLOB_RECURSE IMGCODECS_SRC ${CMAKE_CURRENT_LIST_DIR}/source/imgcodecs/*.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/imgcodecs.hpp)
  endif()
  if(${MNN_CVCORE})
    add_definitions(-DMNN_CVCORE)
    file(GLOB_RECURSE CVCORE_SRC ${CMAKE_CURRENT_LIST_DIR}/source/core/*.cpp ${CMAKE_CURRENT_LIST_DIR}/include/cv/core.hpp)
  endif()

  IF(MNN_SEP_BUILD)
    IF(MNN_BUILD_SHARED_LIBS)
      add_library(MNNOpenCV SHARED ${IMGPROC_SRC} ${CALIB3D_SRC} ${IMGCODECS_SRC} ${CVCORE_SRC})
      target_link_libraries(MNNOpenCV MNN MNN_Express)
    ELSE()
      add_library(MNNOpenCV STATIC ${IMGPROC_SRC} ${CALIB3D_SRC} ${IMGCODECS_SRC} ${CVCORE_SRC})
    ENDIF()
  ELSE()
    add_library(MNNOpenCV OBJECT ${IMGPROC_SRC} ${CALIB3D_SRC} ${IMGCODECS_SRC} ${CVCORE_SRC})
  ENDIF()
  IF(CMAKE_SYSTEM_NAME MATCHES "^Android" AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    IF(NOT NATIVE_INCLUDE_OUTPUT)
      set(NATIVE_INCLUDE_OUTPUT ".")
    ENDIF()
    add_custom_command(
      TARGET MNNOpenCV
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/include ${NATIVE_INCLUDE_OUTPUT}
    )
  ELSE()
    INSTALL(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/cv DESTINATION include FILES_MATCHING PATTERN *.hpp)
  ENDIF()
ENDIF()