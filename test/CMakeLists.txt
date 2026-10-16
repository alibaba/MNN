set(TEST_DEPS ${MNN_DEPS})
if(MNN_VULKAN AND MNN_VULKAN_ARM_NX AND NOT MNN_BUILD_SHARED_LIBS)
  find_package(Vulkan REQUIRED)
  add_executable(vulkan_arm_nx_test.out ${CMAKE_CURRENT_LIST_DIR}/../tools/vulkan/VulkanArmNxTest.cpp)
  target_compile_definitions(vulkan_arm_nx_test.out PRIVATE MNN_VULKAN_ARM_NX)
  if(NOT MNN_USE_SYSTEM_LIB)
    target_compile_definitions(vulkan_arm_nx_test.out PRIVATE MNN_USE_LIB_WRAPPER)
  endif()
  target_include_directories(vulkan_arm_nx_test.out BEFORE PRIVATE ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/../source/backend/vulkan)
  target_link_libraries(vulkan_arm_nx_test.out PRIVATE ${MNN_DEPS} ${Vulkan_LIBRARIES})
  if(MNN_BUILD_LLM)
    add_executable(vulkan_arm_nx_llm_test.out ${CMAKE_CURRENT_LIST_DIR}/../tools/vulkan/VulkanArmNxLlmTest.cpp)
    target_compile_definitions(vulkan_arm_nx_llm_test.out PRIVATE MNN_VULKAN_ARM_NX)
    target_include_directories(vulkan_arm_nx_llm_test.out BEFORE PRIVATE ${Vulkan_INCLUDE_DIRS}
      ${CMAKE_CURRENT_LIST_DIR}/../source/backend/vulkan
      ${CMAKE_CURRENT_LIST_DIR}/../transformers/llm/engine/include)
    target_link_libraries(vulkan_arm_nx_llm_test.out PRIVATE ${MNN_DEPS} ${Vulkan_LIBRARIES})
  endif()
endif()
if(MNN_WITH_PLUGIN)
  include(${CMAKE_CURRENT_LIST_DIR}/plugin/CMakeLists.txt)
  list(APPEND TEST_DEPS plugin_matmul)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "^Android")
  list(APPEND TEST_DEPS android)
endif()

if(APPLE)
  file(GLOB_RECURSE Files ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.mm)
else()
  file(GLOB_RECURSE Files ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
endif()

if(NOT MNN_BUILD_TRAIN)
  file(GLOB_RECURSE GradTestFiles ${CMAKE_CURRENT_LIST_DIR}/grad/*.cpp)
  file(GLOB_RECURSE NNTestFiles ${CMAKE_CURRENT_LIST_DIR}/nn/*.cpp)
  list(REMOVE_ITEM Files ${GradTestFiles} ${NNTestFiles})
endif()

add_executable(run_test.out ${Files})
if(TARGET MNNRVV)
  target_compile_definitions(run_test.out PRIVATE MNN_USE_RVV)
endif()
set_property(SOURCE
  ${CMAKE_CURRENT_LIST_DIR}/backend/riscv/RVVReductionReluTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/RVVMatrixDeconvTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/RVVImageProcessTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/RVVC4AffineTest.cpp
  ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/RVVMatMulTest.cpp
  APPEND PROPERTY
  COMPILE_DEFINITIONS MNN_TEST_RVV_ENABLED=$<BOOL:${MNN_USE_RVV}>)
set_property(SOURCE ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/RVVAbsMaxTest.cpp APPEND PROPERTY
  COMPILE_DEFINITIONS MNN_TEST_RVV_ENABLED=$<BOOL:${MNN_USE_RVV}>)
set_property(SOURCE ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/RVVConvRunLineDepthwiseTest.cpp APPEND PROPERTY
  COMPILE_DEFINITIONS MNN_TEST_RVV_ENABLED=$<BOOL:${MNN_USE_RVV}>)
target_link_libraries(run_test.out ${MNN_DEPS})
if (MSVC)
  target_compile_options(run_test.out PRIVATE /bigobj)
endif()
if (MNN_SUPPORT_BF16)
    target_compile_options(run_test.out PRIVATE -DMNN_SUPPORT_BF16)
endif()
if (NOT MNN_BUILD_SHARED_LIBS)
  if(APPLE)
    set(TEST_DEPS -Wl,-all_load ${TEST_DEPS} -Wl,-noall_load)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(TEST_DEPS -Wl,--whole-archive ${TEST_DEPS} -Wl,--no-whole-archive)
  elseif(MSVC)
    foreach (DEPEND ${TEST_DEPS})
      target_link_options(run_test.out PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
    endforeach ()
  endif()
else()
  target_link_libraries(run_test.out ${TEST_DEPS})
endif()
target_include_directories(run_test.out PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/
    ${CMAKE_CURRENT_LIST_DIR}/../transformers/llm/engine/src/)
if (APPLE)
  find_library(FOUNDATION Foundation REQUIRED)
  target_link_libraries(run_test.out ${FOUNDATION})
endif()
