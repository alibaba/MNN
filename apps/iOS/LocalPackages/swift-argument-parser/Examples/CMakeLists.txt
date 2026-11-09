add_executable(math
  math/Math.swift)
target_compile_options(math PRIVATE
  -parse-as-library)
target_link_libraries(math PRIVATE
  ArgumentParser
  $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Linux>:m>)

add_executable(repeat
  repeat/Repeat.swift)
target_compile_options(repeat PRIVATE
  -parse-as-library)
target_link_libraries(repeat PRIVATE
  ArgumentParser)

add_executable(roll
  roll/main.swift
  roll/SplitMix64.swift)
target_link_libraries(roll PRIVATE
  ArgumentParser)
