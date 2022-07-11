# Environment
## Common Prerequisites
    cmake 3.x
### Mac
    os < 12.3 (see FAQ Q1)
    command line tool < 13.3 (see FAQ Q3)
    Xcode use Legacy Build System (see FAQ Q4)

## Android Build Prerequisites
    android sdk and ndk < r22 (see FAQ Q2), ANDROID_SDK and ANDROID_NDK env var should be set
### Linux (Command line only)
```bash
apt install -y android-sdk # switch mirror when slow, link: https://blog.csdn.net/qq_31456593/article/details/89638163
ANDROID_SDK=/usr/lib/android-sdk
curl -LO https://dl.google.com/android/repository/android-ndk-r21-linux-x86_64.zip
unzip android-ndk-r21-linux-x86_64.zip -d $ANDROID_SDK 1>/dev/null
echo -e "\nexport ANDROID_SDK=$ANDROID_SDK\nexport ANDROID_NDK=$ANDROID_SDK/android-ndk-r21" >> ~/.bashrc
```
### Other
Install Android Studio, download ndk 21.x in SDK Manager

## iOS Build Prerequisites
    ruby from rbenv # avoid abuse root permission
### Mac
```bash
brew install rbenv  # avoid abuse root permission
rbenv install 3.1.2
rbenv global 3.1.2
echo -e "\nexport PATH=\"$HOME/.rbenv/bin:$PATH\"\neval \"$(rbenv init -)\"" | tee -a ~/.bash_profile ~/.zshrc
```

# Benchmark
## Build bench tools
### PC
```bash
./build.sh --pc --mnn-tag 90719ce2 --torch-tag v1.11.0 --tf-tag v2.7.0
```
### Android
```bash
./build.sh --android --mnn-tag 90719ce2 --torch-tag v1.11.0 --tf-tag v2.7.0
```
### iOS
```bash
./build.sh --ios --mnn-tag 90719ce2 --torch-tag v1.11.0 --tf-tag v2.7.0
```
## Convert Models (onnx -> mnn/pb/tflite/torch/torchlite)
```bash
./convert.sh --mnn-tag 90719ce2 --torch-tag v1.11.0
```
## Run bench
### PC
```bash
./bench.sh --pc --mnn --torch --tf --sync-models --tooldir dist
```
### Android
```bash
./bench.sh --android --mnn --torch --tf --sync-models --tooldir dist
```
### iOS
```bash
./bench.sh --ios --mnn --torch --tf --sync-models --tooldir dist
```

# FAQ
<b>Q1: build tensorflow/tflite for ios failed: env: python: no such file or directory</b>  
<b>A</b>: Some bazel version can't deal with --incompatible_strict_action_env and --action_env correctly when run py_binary (bundletool on macos), then it use /usr/bin/python (instead of /usr/local/bin/python from homebrew). Mac >= 12.3 remove python 2.7 from /usr/bin, then bundletool run failed. 
  Solution: Downgrade os version to 12.2, 
            Or change shebang line (#!/usr/bin/env python -> #!/usr/bin/env python3) of failed file (tensorflow/bazel-out/host/bin/external/build_bazel_rules_apple/tools/bundletool/bundletool)
---
<b>Q2: No such file or directory: '{ANDROID_NDK}/platforms'</b>  
<b>A</b>: older toolchain (include platforms) be removed on android ndk r22, which breaking tflite compile, so ndk should be lower than r22
---
<b>Q3: section \_\_TEXT/\_\_const address out of range for architecture x86_64</b>
<b>A</b>: pytorch compile failed on command line tool 13.3, see issue:  https://github.com/pytorch/pytorch/issues/76094#issuecomment-1106624976
---
<b>Q4: Building for iOS, but the linked and embedded framework 'TensorFlowLiteBenchmarkC.framework' was built for iOS + iOS Simulator.</b>
<b>A</b>: tflite produce ios\_fat framework containing many arch, which isn't support on New Build System (need XCFramework)
