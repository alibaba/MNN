#!/usr/bin/env bash

set -ex

if [ ! -d ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 ]; then
  echo "Please download the pre-trained model for testing."
  echo "You can refer to"
  echo ""
  echo "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english"
  echo "for help"
  exit 1
fi

if [[ ! -f ../build/lib/libsherpa-onnx-c-api.a && ! -f ../build/lib/libsherpa-onnx-c-api.dylib  && ! -f ../build/lib/libsherpa-onnx-c-api.so ]]; then
  echo "Please build sherpa-onnx first. You can use"
  echo ""
  echo "  cd /path/to/sherpa-onnx"
  echo "  mkdir build"
  echo "  cd build"
  echo "  cmake .."
  echo "  make -j4"
  exit 1
fi

if [ ! -f ./decode-file-c-api ]; then
  make
fi

./decode-file-c-api \
  --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
  --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
  --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
  --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
  ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav

# Run with hotwords

echo "礼 拜 二" > hotwords.txt

./decode-file-c-api \
  --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
  --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
  --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
  --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
  --hotwords-file=hotwords.txt \
  --hotwords-score=1.5 \
  --decoding-method=modified_beam_search \
  ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav
