#!/bin/bash

pushd "$(dirname $0)"/../.. > /dev/null
pushd resource > /dev/null

# mkdir
if [ ! -d build ]; then
  mkdir build
fi

# build converter
CONVERTER=build/MNNConvert
if [ ! -e $CONVERTER ]; then
  echo "building converter ..."
  pushd ../tools/converter > /dev/null
  ./build_tool.sh
  popd > /dev/null
  cp ../tools/converter/build/MNNConvert $CONVERTER
fi

# functions
get_caffe1() { # model_URL, model_path, prototxt_URL, prototxt_path, model, MNN_path
  if [ ! -e $6 ]; then
    ([ ! -e $2 ] || [ ! -e $4 ]) && echo "downloading model for $5"
    ([ ! -e $2 ] && curl -s $1 > $2) &
    ([ ! -e $4 ] && curl -s $3 > $4) &
    wait
    ./$CONVERTER -f CAFFE --modelFile $2 --prototxt $4 --MNNModel $6 --bizCode 0000
  fi
}

get_tensorflow_lite() {
  if [ ! -e $4 ]; then
    pushd build > /dev/null
    [ ! -e $2.tgz ] && echo "downloading model for $3"  && curl -s $1 > $2.tgz
    tar -xzf $2.tgz $2
    popd > /dev/null
    ./$CONVERTER -f TFLITE --modelFile build/$2 --MNNModel $4 --bizCode 0000
  fi
}

get_portrait_lite() {
    if [ ! -e $4 ]; then
        pushd build > /dev/null
        [ ! -e $2 ] && echo "downloading model for $3"  && curl -s $1 > $2
        popd > /dev/null
        ./$CONVERTER -f TFLITE --modelFile build/$2 --MNNModel $4 --bizCode 0000
    fi
}

# get models

## Using MobileNet V1 downloaded from: https://github.com/shicai/MobileNet-Caffe/
get_caffe1 \
  "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet.caffemodel" \
  "build/mobilenet_v1.caffe.caffemodel" \
  "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_deploy.prototxt" \
  "build/mobilenet_v1.caffe.prototxt" \
  "MobileNet V1" \
  "model/MobileNet/v1/mobilenet_v1.caffe.mnn"

## Using MobileNet V2 downloaded from: https://github.com/shicai/MobileNet-Caffe/
get_caffe1 \
  "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.caffemodel" \
  "build/mobilenet_v2.caffe.caffemodel" \
  "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt" \
  "build/mobilenet_v2.caffe.prototxt" \
  "MobileNet V2" \
  "model/MobileNet/v2/mobilenet_v2.caffe.mnn"

## Using SqueezeNet V1.0 downloaded from: https://github.com/DeepScale/SqueezeNet/
get_caffe1 \
"https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel" \
"build/squeezenet_v1.0.caffe.caffemodel" \
"https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.0/deploy.prototxt" \
"build/squeezenet_v1.0.caffe.prototxt" \
"SqueezeNet V1.0" \
"model/SqueezeNet/v1.0/squeezenet_v1.0.caffe.mnn"

## Using SqueezeNet V1.1 downloaded from: https://github.com/DeepScale/SqueezeNet/
get_caffe1 \
"https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel" \
"build/squeezenet_v1.1.caffe.caffemodel" \
"https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt" \
"build/squeezenet_v1.1.caffe.prototxt" \
"SqueezeNet V1.1" \
"model/SqueezeNet/v1.1/squeezenet_v1.1.caffe.mnn"

## Using MobileNet V2 downloaded from: http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
get_tensorflow_lite \
  "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz" \
  "mobilenet_v2_1.0_224.tflite" \
  "MobileNet V2 TFLite" \
  "model/MobileNet/v2/mobilenet_v2_1.0_224.tflite.mnn"

## Using MobileNet V2 downloaded from: http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
get_tensorflow_lite \
  "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz" \
  "mobilenet_v2_1.0_224_quant.tflite" \
  "MobileNet V2 TFLite Quantized" \
  "model/MobileNet/v2/mobilenet_v2_1.0_224_quant.tflite.mnn"

## Using deeplab v3 downloaded from: https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite
get_portrait_lite \
"https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite" \
"deeplabv3_257_mv_gpu.tflite" \
"deeplabv3" \
"model/Portrait/Portrait.tflite.mnn"

popd > /dev/null
popd > /dev/null
