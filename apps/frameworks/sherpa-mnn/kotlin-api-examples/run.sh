#!/usr/bin/env bash
#
# This scripts shows how to build JNI libs for sherpa-onnx
# Note: This scripts runs only on Linux and macOS, though sherpa-onnx
# supports building JNI libs for Windows.

set -ex

if [[ ! -f ../build/lib/libsherpa-onnx-jni.dylib  && ! -f ../build/lib/libsherpa-onnx-jni.so ]]; then
  mkdir -p ../build
  pushd ../build
  cmake \
    -DSHERPA_MNN_ENABLE_PYTHON=OFF \
    -DSHERPA_MNN_ENABLE_TESTS=OFF \
    -DSHERPA_MNN_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_MNN_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_MNN_ENABLE_JNI=ON \
    ..

  make -j4
  ls -lh lib
  popd
fi

export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

function testSpeakerEmbeddingExtractor() {
  if [ ! -f ./3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx
  fi

  if [ ! -f ./speaker1_a_cn_16k.wav ]; then
    curl -SL -O https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_a_cn_16k.wav
  fi

  if [ ! -f ./speaker1_b_cn_16k.wav ]; then
    curl -SL -O https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_b_cn_16k.wav
  fi

  if [ ! -f ./speaker2_a_cn_16k.wav ]; then
    curl -SL -O https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker2_a_cn_16k.wav
  fi

  out_filename=test_speaker_id.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_speaker_id.kt \
    OnlineStream.kt \
    Speaker.kt \
    SpeakerEmbeddingExtractorConfig.kt \
    WaveReader.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}


function testOnlineAsr() {
  if [ ! -f ./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt ]; then
    git lfs install
    GIT_CLONE_PROTECTION_ACTIVE=false git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21
  fi

  if [ ! -f ./sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms.tar.bz2
    tar xvf sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms.tar.bz2
    rm sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms.tar.bz2
  fi

  if [ ! -d ./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13 ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
    tar xvf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
    rm sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
  fi

  if [ ! -d ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18 ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
    tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
    rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  fi

  out_filename=test_online_asr.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_online_asr.kt \
    FeatureConfig.kt \
    OnlineRecognizer.kt \
    OnlineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}

function testTts() {
  if [ ! -f ./vits-piper-en_US-amy-low/en_US-amy-low.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
    tar xf vits-piper-en_US-amy-low.tar.bz2
    rm vits-piper-en_US-amy-low.tar.bz2
  fi

  if [ ! -f ./matcha-icefall-zh-baker/model-steps-3.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
    tar xvf matcha-icefall-zh-baker.tar.bz2
    rm matcha-icefall-zh-baker.tar.bz2
  fi

  if [ ! -f ./hifigan_v2.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx
  fi

  if [ ! -f ./kokoro-multi-lang-v1_0/model.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
    tar xf kokoro-multi-lang-v1_0.tar.bz2
    rm kokoro-multi-lang-v1_0.tar.bz2
  fi

  if [ ! -f ./kokoro-en-v0_19/model.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
    tar xf kokoro-en-v0_19.tar.bz2
    rm kokoro-en-v0_19.tar.bz2
  fi

  out_filename=test_tts.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_tts.kt \
    Tts.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}


function testAudioTagging() {
  if [ ! -d sherpa-onnx-zipformer-audio-tagging-2024-04-09 ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
    tar xvf sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
    rm sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
  fi

  out_filename=test_audio_tagging.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_audio_tagging.kt \
    AudioTagging.kt \
    OfflineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}


function testSpokenLanguageIdentification() {
  if [ ! -f ./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
    tar xvf sherpa-onnx-whisper-tiny.tar.bz2
    rm sherpa-onnx-whisper-tiny.tar.bz2
  fi

  if [ ! -f ./spoken-language-identification-test-wavs/ar-arabic.wav ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2
    tar xvf spoken-language-identification-test-wavs.tar.bz2
    rm spoken-language-identification-test-wavs.tar.bz2
  fi

  out_filename=test_language_id.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_language_id.kt \
    SpokenLanguageIdentification.kt \
    OfflineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}

function testOfflineAsr() {
  if [ ! -f ./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
    tar xvf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
    rm sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
    ls -lh sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16
  fi

  if [ ! -f ./sherpa-onnx-moonshine-tiny-en-int8/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
    tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
    rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
  fi

  if [ ! -f ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
    tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
    rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  fi

  if [ ! -f ./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
    tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
    rm sherpa-onnx-whisper-tiny.en.tar.bz2
  fi

  if [ ! -f ./sherpa-onnx-nemo-ctc-en-citrinet-512/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
    tar xvf sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
    rm sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
  fi

  if [ ! -f ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
    tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
    rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  fi

  if [ ! -f ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2
    tar xvf sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2
    rm sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2
  fi

  out_filename=test_offline_asr.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_offline_asr.kt \
    FeatureConfig.kt \
    OfflineRecognizer.kt \
    OfflineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt

  ls -lh $out_filename
  java -Djava.library.path=../build/lib -jar $out_filename
}

function testInverseTextNormalizationOfflineAsr() {
  if [ ! -f ./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
    tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
    rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  fi

  if [ ! -f ./itn-zh-number.wav ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
  fi

  if [ ! -f ./itn_zh_number.fst ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
  fi

  out_filename=test_itn_offline_asr.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_itn_offline_asr.kt \
    FeatureConfig.kt \
    OfflineRecognizer.kt \
    OfflineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt

  ls -lh $out_filename
  java -Djava.library.path=../build/lib -jar $out_filename
}

function testInverseTextNormalizationOnlineAsr() {
  if [ ! -f ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
    tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
    rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
  fi

  if [ ! -f ./itn-zh-number.wav ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
  fi

  if [ ! -f ./itn_zh_number.fst ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
  fi

  out_filename=test_itn_online_asr.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_itn_online_asr.kt \
    FeatureConfig.kt \
    OnlineRecognizer.kt \
    OnlineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt

  ls -lh $out_filename
  java -Djava.library.path=../build/lib -jar $out_filename
}

function testOfflinePunctuation() {
  if [ ! -f ./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
    tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
    rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
  fi

  out_filename=test_offline_punctuation.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    ./test_offline_punctuation.kt \
    ./OfflinePunctuation.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}

function testOnlinePunctuation() {
  if [ ! -f ./sherpa-onnx-online-punct-en-2024-08-06/model.int8.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
    tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
    rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
  fi

  out_filename=test_online_punctuation.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    ./test_online_punctuation.kt \
    ./OnlinePunctuation.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}

function testOfflineSpeakerDiarization() {
  if [ ! -f ./sherpa-onnx-pyannote-segmentation-3-0/model.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
    tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
    rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  fi

  if [ ! -f ./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
  fi

  if [ ! -f ./0-four-speakers-zh.wav ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav
  fi

  out_filename=test_offline_speaker_diarization.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_offline_speaker_diarization.kt \
    OfflineSpeakerDiarization.kt \
    Speaker.kt \
    SpeakerEmbeddingExtractorConfig.kt \
    OnlineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}

testOfflineSpeakerDiarization
testSpeakerEmbeddingExtractor
testOnlineAsr
testTts
testAudioTagging
testSpokenLanguageIdentification
testOfflineAsr
testOfflinePunctuation
testOnlinePunctuation
testInverseTextNormalizationOfflineAsr
testInverseTextNormalizationOnlineAsr
