#!/bin/bash
set -e

usage() {
    echo "Usage: $0 --mnn-tag MNN_TAG [--torch-tag TORCH_TAG]"
    echo -e "\t--mnn_tag mnn git tag"
    echo -e "\t--torch-tag torch git tag"
    exit 1
}

while getopts ":h-:" opt; do
  case "$opt" in
    -)
        case "$OPTARG" in
            mnn-tag) MNN_TAG=${!OPTIND} ; OPTIND=$(( $OPTIND + 1 )) ;;
            torch-tag) TORCH_TAG=${!OPTIND} ; OPTIND=$(( $OPTIND + 1 )) ;;
            *) usage ;;
        esac ;;
    h|? ) usage ;;
  esac
done
# Convert onnx -> mnn/pt/ptlite/pb/tflite
rm -rf onnx-pytorch
git clone --depth 1 --branch feature/mobilenetv2_bert_support https://github.com/yutianhang/onnx-pytorch.git
python -m pip uninstall -y onnx-pytorch
pushd onnx-pytorch
python -m pip install -r requirements.txt
python -m pip install .
popd

python -m pip install onnx_tf tensorflow==2.7.0 tensorflow-probability==0.15.0 onnx

#MNN_REPO=https://github.com/alibaba/MNN.git
MNN_REPO=git@gitlab.alibaba-inc.com:AliNN/AliNNPrivate.git
if [ -z "$MNN_TAG" ]; then
    echo "MNNConvert need MNN git repo tag --mnn-tag MNN_TAG"
    exit 1
fi
if [ ! -d MNN ] || ([ "$(cd MNN && git describe --tags)" != "$MNN_TAG" ] && [[ "$(cd MNN && git rev-parse HEAD)" != "${MNN_TAG}"* ]]); then
    rm -rf MNN
    set +e
    git clone --depth 1 --branch $MNN_TAG $MNN_REPO MNN
    set -e
    if [ ! -d MNN ]; then
        git clone $MNN_REPO MNN
        pushd MNN
        git checkout $MNN_TAG
        popd
    fi
fi
if [ ! -f MNN/build_converter/MNNConvert ]; then
    pushd MNN
    rm -rf build_converter && mkdir build_converter && pushd build_converter
    cmake -DMNN_BUILD_CONVERTER=ON ..
    make -j
    popd
    popd
fi

if [ ! -d pytorch ] || [ "$(cd pytorch && git describe --tags)" != "$TORCH_TAG" ]; then
    rm -rf pytorch
    git clone --depth 1 --branch $TORCH_TAG https://github.com/pytorch/pytorch.git
    pushd pytorch
    git submodule sync --recursive
    git submodule update --init --recursive
    popd
fi
python -m pip uninstall -y torch
pushd pytorch
USE_PYTORCH_METAL=ON USE_VULKAN=ON python setup.py install
popd
python convert.py --modeldir models
