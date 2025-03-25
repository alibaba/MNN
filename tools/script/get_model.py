#!/usr/bin/python
import sys
import urllib
import os
CONVERTER = os.path.join("build", "MNNConvert")
print("Converter Path:" + CONVERTER)

def download(url, dest):
    if os.path.exists(dest):
        print(dest + " exists, skip")
        return
    print("Download " + url + " -> " + dest)
    urllib.urlretrieve(url, dest)


def get_caffe1(urlmodel, destmodel, urlproto, destproto, name, position):
    download(urlmodel, destmodel)
    download(urlproto, destproto)
    print("Caffe Dest: " + position)
    print(os.popen(CONVERTER + " -f CAFFE --modelFile " + destmodel + " --prototxt " + destproto + " --MNNModel " + position + " --bizCode 0000 --keepInputFormat=0").read())

def get_tensorflow_lite(urlmodel, destmodel, name, position):
    download(urlmodel, destmodel)
    print("Tflite Dest: " + position)
    print(os.popen(CONVERTER + " -f TFLITE --modelFile " + destmodel + " --MNNModel " + position + " --bizCode 0000 --keepInputFormat=0").read())

# get models


## Using SqueezeNet V1.0 downloaded from: https://github.com/DeepScale/SqueezeNet/
get_caffe1(
  "https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel" ,
  "build/squeezenet_v1.0.caffe.caffemodel" ,
  "https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.0/deploy.prototxt" ,
  "build/squeezenet_v1.0.caffe.prototxt" ,
  "SqueezeNet V1.0" ,
  "resource/model/SqueezeNet/v1.0/squeezenet_v1.0.caffe.mnn")

## Using SqueezeNet V1.1 downloaded from: https://github.com/DeepScale/SqueezeNet/
get_caffe1(
  "https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel" ,
  "build/squeezenet_v1.1.caffe.caffemodel" ,
  "https://raw.githubusercontent.com/DeepScale/SqueezeNet/b6b5ae2ce884a3866c21efd31e103defde8631ae/SqueezeNet_v1.1/deploy.prototxt" ,
  "build/squeezenet_v1.1.caffe.prototxt" ,
  "SqueezeNet V1.1" ,
  "resource/model/SqueezeNet/v1.1/squeezenet_v1.1.caffe.mnn")

## Using MobileNet V2 downloaded from: http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
get_tensorflow_lite(
  "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz" ,
  "build/mobilenet_v2_1.0_224.tflite" ,
  "MobileNet V2 TFLite" ,
  "resource/model/MobileNet/v2/mobilenet_v2_1.0_224.tflite.mnn")

## Using MobileNet V2 downloaded from: http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
get_tensorflow_lite(
  "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz" ,
  "mobilenet_v2_1.0_224_quant.tflite" ,
  "MobileNet V2 TFLite Quantized" ,
  "resource/model/MobileNet/v2/mobilenet_v2_1.0_224_quant.tflite.mnn")

## Using deeplab v3 downloaded from: https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite
get_tensorflow_lite(
  "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite" ,
  "build/deeplabv3_257_mv_gpu.tflite" ,
  "deeplabv3" ,
  "resource/model/Portrait/Portrait.tflite.mnn")

## Using MobileNet V1 downloaded from: https://github.com/shicai/MobileNet-Caffe/
get_caffe1(
  "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet.caffemodel" ,
  "build/mobilenet_v1.caffe.caffemodel" ,
  "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_deploy.prototxt" ,
  "build/mobilenet_v1.caffe.prototxt" ,
  "MobileNet V1" ,
  "resource/model/MobileNet/v1/mobilenet_v1.caffe.mnn")
