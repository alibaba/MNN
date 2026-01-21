{
  "targets": [
    {
      "target_name": "mnn",
      "sources": [
        "src/mnn_node.cc",
        "src/interpreter.cc",
        "src/session.cc",
        "src/tensor.cc",
        "src/utils.cc"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<(module_root_dir)/../include",
        "<(module_root_dir)/../express",
        "<(module_root_dir)/../3rd_party/flatbuffers/include"
      ],
      "libraries": [
        "<(module_root_dir)/../build/libMNN.so"
      ],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "cflags_cc": ["-std=c++11"],
      "defines": ["NAPI_CPP_EXCEPTIONS"],
      "conditions": [
        [
          "OS=='mac'",
          {
            "libraries": ["<(module_root_dir)/../build/libMNN.dylib"],
            "xcode_settings": {
              "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
              "CLANG_CXX_LIBRARY": "libc++",
              "MACOSX_DEPLOYMENT_TARGET": "10.9",
              "OTHER_LDFLAGS": ["-Wl,-rpath,@loader_path/../build"]
            }
          }
        ],
        [
          "OS=='win'",
          {
            "libraries": ["<(module_root_dir)/../build/MNN.lib"],
            "msvs_settings": {
              "VCCLCompilerTool": {
                "ExceptionHandling": 1
              }
            }
          }
        ]
      ]
    }
  ]
}
