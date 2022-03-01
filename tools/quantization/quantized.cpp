//
//  quantized.cpp
//  MNN
//
//  Created by MNN on 2019/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <sstream>
#include <string>
#include "calibration.hpp"
#include "logkit.h"

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        DLOG(INFO) << "Usage: ./quantized.out src.mnn dst.mnn preTreatConfig.json\n";
        return 0;
    }
    const char* modelFile      = argv[1];
    const char* preTreatConfig = argv[3];
    const char* dstFile        = argv[2];
    DLOG(INFO) << ">>> modelFile: " << modelFile;
    DLOG(INFO) << ">>> preTreatConfig: " << preTreatConfig;
    DLOG(INFO) << ">>> dstFile: " << dstFile;
    std::unique_ptr<MNN::NetT> netT;
    {
        //std::ifstream input(modelFile);
        std::ifstream input(modelFile, std::ifstream::in | std::ifstream::binary);
        std::ostringstream outputOs;
        outputOs << input.rdbuf();
        netT = MNN::UnPackNet(outputOs.str().c_str());
    }

    // temp build net for inference
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = MNN::Net::Pack(builder, netT.get());
    builder.Finish(offset);
    int size      = builder.GetSize();
    auto ocontent = builder.GetBufferPointer();

    // model buffer for creating mnn Interpreter
    std::unique_ptr<uint8_t> modelForInference(new uint8_t[size]);
    memcpy(modelForInference.get(), ocontent, size);

    std::unique_ptr<uint8_t> modelOriginal(new uint8_t[size]);
    memcpy(modelOriginal.get(), ocontent, size);

    netT.reset();
    netT = MNN::UnPackNet(modelOriginal.get());

    // quantize model's weight
    DLOG(INFO) << "Calibrate the feature and quantize model...";
    std::shared_ptr<Calibration> calibration(
        new Calibration(netT.get(), modelForInference.get(), size, preTreatConfig, std::string(modelFile), std::string(dstFile)));
    calibration->runQuantizeModel();
    calibration->dumpTensorScales(dstFile);
    DLOG(INFO) << "Quantize model done!";

    return 0;
}
