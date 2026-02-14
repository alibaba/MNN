# Sana Diffusion Android Integration Progress

Status: **Completed (Native & Kotlin Hooked)**  
Date: 2026-01-20

## 1. Overview
The Sana Diffusion model has been integrated into the `apps/Android/MnnLlmChat` application. Unlike standard Stable Diffusion, Sana requires a two-stage process: Text-to-LLM-Features (via `SanaLlm`) and Features-to-Image (via `Diffusion` type 2). This integration allows Sana models to be recognized and run as local models from `/data/local/tmp/mnn_models/`.

## 2. Key Changes

### Native Layer (C++)
- **`sana_session.h/cpp`**: New wrapper class `mls::SanaSession`.
    - Manages `MNN::DIFFUSION::SanaLlm` for text processing.
    - Manages `MNN::DIFFUSION::Diffusion` (Type 2) for image generation.
    - Hardcoded to use `MNN_FORWARD_OPENCL` for optimal Android performance.
- **`sana_jni.cpp`**: JNI implementation for `com.alibaba.mnnllm.android.llm.SanaSession`.
    - `initNative`: Instantiates and loads both LLM and Diffusion models.
    - `generateNative`: Orchestrates the two-stage inference and handles progress callbacks.
- **`CMakeLists.txt`**: Added `sana_session.cpp` and `sana_jni.cpp` to `mnnllmapp` library.

### Kotlin Layer (Android)
- **`SanaSession.kt`**: Implements `ChatSession` interface.
    - Uses JNI to communicate with the native Sana implementation.
    - Handles `diffusion_memory_mode` from App Settings.
- **`ModelTypeUtils.kt`**: 
    - Added `isSanaModel(name)` detection (checks for "sana" keyword).
    - Updated `isDiffusionModel(name)` to include Sana, ensuring correct UI behavior (e.g., hiding LLM-specific switches).
- **`ChatService.kt`**:
    - Updated `createSession` factory to return `SanaSession` when a Sana model is identified.
- **`ChatPresenter.kt`**:
    - Optimized default generation steps: **5 steps** for Sana (vs 20 for standard Diffusion).

## 3. Model Structure Requirement
The local model folder in `/data/local/tmp/mnn_models/` should follow this structure:
```text
your_sana_model/
├── config.json           # Basic model info
├── llm/
│   ├── config.json       # Qwen model config
│   └── meta_queries.mnn  # Sana meta queries
├── vae/                  # VAE model files
└── transformer/          # Sana transformer model files
```

## 4. Verification & Testing
- [x] Native code compiles into `libmnnllmapp.so`.
- [x] JNI package names match Kotlin class paths.
- [x] Local model scanning recognizes "sana" folders.
- [x] `ChatRouter` successfully routes to `SanaSession`.
- [x] Steps count correctly adjusted for Sana.

## 5. Next Steps
- Verify peak memory usage on mid-range devices (Sana can be memory intensive).
- Consider adding a configurable "Steps" slider in the UI for Diffusion models.
