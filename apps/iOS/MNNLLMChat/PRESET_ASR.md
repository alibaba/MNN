# Preset X-ASR demo assets

The preset-audio path runs X-ASR through the in-tree `sherpa-mnn` runtime,
then submits the recognized text to Qwen3.5-2B. Model weights, the sample
recording, and generated static libraries are deliberately kept out of Git.

## Internal test package

Request `mnn-ios-xasr-int8-test-assets-v1.zip` through the approved internal
artifact channel. Do not upload this package to a public pull request or
release until the model and recording redistribution rights are confirmed.

Extract the archive at the repository root:

```bash
ditto -x -k mnn-ios-xasr-int8-test-assets-v1.zip /path/to/MNN
cd /path/to/MNN
shasum -a 256 -c apps/iOS/MNNLLMChat/PRESET_ASR_ASSETS.sha256
```

The archive installs the following ignored files:

```text
apps/iOS/MNNLLMChat/
├── libsherpa-mnn.a
└── MNNLLMiOS/LocalModel/
    ├── preset_audio/audio (1).wav
    └── xasr-mnn-int8/
        ├── encoder-160ms.mnn
        ├── decoder-160ms.mnn
        ├── joiner-160ms.mnn
        └── tokens.txt
```

`libsherpa-mnn.a` in package v1 is an arm64 iOS archive. A universal local
replacement can be generated from `apps/frameworks/sherpa-mnn/build-ios.sh`
and linked through an XCFramework in the Xcode project.

## Build and silent verification

Open `apps/iOS/MNNLLMChat/MNNLLMiOS.xcodeproj`, select the `MNNLLMiOS`
scheme and a signed iOS device, then build and run. The X-ASR preset appears
only when the sample recording is present.

To exercise the production ASR wrapper without opening the microphone or
playing audio, launch the app with:

```text
--preset-asr-silent-probe
```

The probe writes `qwen-xasr-device-probe.json` to the app Documents directory.
For the supplied 2.136875-second Mandarin sample, the expected normalized
transcript is:

```text
给我介绍一下杭州市的美食
```

The UI reports ASR inference time, audio duration, RTF, and real-time multiple.
Model initialization time is logged for diagnostics but intentionally omitted
from the chat bubble.
