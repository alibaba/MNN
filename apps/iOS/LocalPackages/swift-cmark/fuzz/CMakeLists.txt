foreach(fuzzer fuzz_quadratic fuzz_quadratic_brackets)
  add_executable(${fuzzer}
    ${fuzzer}.c)
  target_compile_options(${fuzzer} PRIVATE
    -fsanitize=fuzzer)
  target_link_options(${fuzzer} PRIVATE
    -fsanitize=fuzzer)
  target_link_libraries(${fuzzer} PRIVATE
    libcmark-gfm
    libcmark-gfm-extensions)
endforeach()
