// Build script for MNN Rust binding
// This script compiles the C FFI wrapper and links with MNN library

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");

    let mnn_root = PathBuf::from(&manifest_dir); // Create PathBuf first
    let mnn_root = mnn_root
        .parent()
        .expect("Cannot find MNN root directory")
        .to_path_buf(); // Convert to owned PathBuf

    let mnn_build_dir = mnn_root.join("build");

    // Check if MNN build directory exists
    if !mnn_build_dir.exists() {
        panic!(
            "MNN build directory not found at {:?}. \
             Please build MNN first:\n  cd {:?} && cmake .. && make -j8",
            mnn_build_dir, mnn_root
        );
    }

    // Print cargo instructions
    println!("cargo:rerun-if-changed=csrc/mnn_c.cpp");
    println!("cargo:rerun-if-changed=csrc/mnn_c.h");

    // Compile only the C FFI wrapper
    // The LLM engine is already compiled into libMNN.dylib
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .file("csrc/mnn_c.cpp")
        .include("csrc")
        .include(mnn_root.join("include"))
        .include(mnn_root.join("transformers/llm/engine/include"))
        .include(mnn_root.join("transformers/llm/engine/src"))
        .include(mnn_root.join("express"))
        .include(mnn_root.join("3rd_party/flatbuffers/include"))
        .include(mnn_root.join("3rd_party/half"))
        .include(mnn_root.join("3rd_party"))
        .include(mnn_root.join("3rd_party/httplib"))
        .include(mnn_root.join("tools"))
        .include(mnn_root.join("source"))
        .include(mnn_root.join("schema/current"))
        .flag("-std=c++17")
        .flag("-DLLM_USE_MINJA")
        .flag("-DMNN_BUILD_LLM")
        .warnings(false)
        .opt_level(2);

    // Platform-specific settings
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    match target_os.as_str() {
        "macos" => {
            build.flag("-stdlib=libc++");
        }
        "linux" => {
            // Linux-specific flags
        }
        "windows" => {
            build.flag("/EHsc");
        }
        _ => {}
    }

    build.compile("mnn_c");

    // Link with MNN library
    println!("cargo:rustc-link-search=native={}", mnn_build_dir.display());

    match target_os.as_str() {
        "macos" => {
            println!("cargo:rustc-link-lib=static=MNN");
            println!("cargo:rustc-link-lib=c++");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=CoreGraphics");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=static=MNN");
            println!("cargo:rustc-link-lib=stdc++");
        }
        "windows" => {
            println!("cargo:rustc-link-lib=static=MNN");
        }
        _ => {
            println!("cargo:rustc-link-lib=static=MNN");
        }
    }
}
