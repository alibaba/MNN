//
//  MnistDataset.cpp
//  MNN
//
//  Created by MNN on 2019/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MnistDataset.hpp"
#include <string.h>
#include <fstream>
#include <string>
namespace MNN {
namespace Train {

// referenced from pytorch C++ frontend mnist.cpp
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp
const int32_t kTrainSize          = 60000;
const int32_t kTestSize           = 10000;
const int32_t kImageMagicNumber   = 2051;
const int32_t kTargetMagicNumber  = 2049;
const int32_t kImageRows          = 28;
const int32_t kImageColumns       = 28;
const char* kTrainImagesFilename  = "train-images-idx3-ubyte";
const char* kTrainTargetsFilename = "train-labels-idx1-ubyte";
const char* kTestImagesFilename   = "t10k-images-idx3-ubyte";
const char* kTestTargetsFilename  = "t10k-labels-idx1-ubyte";

bool check_is_little_endian() {
    const uint32_t word = 1;
    return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
}

constexpr uint32_t flip_endianness(uint32_t value) {
    return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) | ((value & 0xff0000u) >> 8u) |
           ((value & 0xff000000u) >> 24u);
}

uint32_t read_int32(std::ifstream& stream) {
    static const bool is_little_endian = check_is_little_endian();
    uint32_t value;
    stream.read(reinterpret_cast<char*>(&value), sizeof value);
    return is_little_endian ? flip_endianness(value) : value;
}

uint32_t expect_int32(std::ifstream& stream, uint32_t expected) {
    const auto value = read_int32(stream);
    // clang-format off
    MNN_ASSERT(value == expected);
    // clang-format on
    return value;
}

std::string join_paths(std::string head, const std::string& tail) {
    if (head.back() != '/') {
        head.push_back('/');
    }
    head += tail;
    return head;
}

VARP read_images(const std::string& root, bool train) {
    const auto path = join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
    std::ifstream images(path, std::ios::binary);
    if (!images.is_open()) {
        MNN_PRINT("Error opening images file at %s", path.c_str());
        MNN_ASSERT(false);
    }

    const auto count = train ? kTrainSize : kTestSize;

    // From http://yann.lecun.com/exdb/mnist/
    expect_int32(images, kImageMagicNumber);
    expect_int32(images, count);
    expect_int32(images, kImageRows);
    expect_int32(images, kImageColumns);

    std::vector<int> dims = {count, 1, kImageRows, kImageColumns};
    int length            = 1;
    for (int i = 0; i < dims.size(); ++i) {
        length *= dims[i];
    }
    auto data = _Input(dims, NCHW, halide_type_of<uint8_t>());
    images.read(reinterpret_cast<char*>(data->writeMap<uint8_t>()), length);
    return data;
}

VARP read_targets(const std::string& root, bool train) {
    const auto path = join_paths(root, train ? kTrainTargetsFilename : kTestTargetsFilename);
    std::ifstream targets(path, std::ios::binary);
    if (!targets.is_open()) {
        MNN_PRINT("Error opening images file at %s", path.c_str());
        MNN_ASSERT(false);
    }

    const auto count = train ? kTrainSize : kTestSize;

    expect_int32(targets, kTargetMagicNumber);
    expect_int32(targets, count);

    std::vector<int> dims = {count};
    int length            = 1;
    for (int i = 0; i < dims.size(); ++i) {
        length *= dims[i];
    }
    auto labels = _Input(dims, NCHW, halide_type_of<uint8_t>());
    targets.read(reinterpret_cast<char*>(labels->writeMap<uint8_t>()), length);

    return labels;
}

MnistDataset::MnistDataset(const std::string root, Mode mode)
    : mImages(read_images(root, mode == Mode::TRAIN)), mLabels(read_targets(root, mode == Mode::TRAIN)) {
    mImagePtr  = mImages->readMap<uint8_t>();
    mLabelsPtr = mLabels->readMap<uint8_t>();
}

Example MnistDataset::get(size_t index) {
    auto data  = _Input({1, kImageRows, kImageColumns}, NCHW, halide_type_of<uint8_t>());
    auto label = _Input({}, NCHW, halide_type_of<uint8_t>());

    auto dataPtr = mImagePtr + index * kImageRows * kImageColumns;
    ::memcpy(data->writeMap<uint8_t>(), dataPtr, kImageRows * kImageColumns);

    auto labelPtr = mLabelsPtr + index;
    ::memcpy(label->writeMap<uint8_t>(), labelPtr, 1);

    auto returnIndex = _Const(index);
    // return the index for test
    return {{data, returnIndex}, {label}};
}

size_t MnistDataset::size() {
    return mImages->getInfo()->dim[0];
}

const VARP MnistDataset::images() {
    return mImages;
}

const VARP MnistDataset::labels() {
    return mLabels;
}

DatasetPtr MnistDataset::create(const std::string path, Mode mode) {
    DatasetPtr res;
    res.mDataset.reset(new MnistDataset(path, mode));
    return res;
}
}
}
