//
//  quantized.cpp
//  MNN
//
//  Created by MNN on 2019/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <sstream>
#include "calibration.hpp"
#include "logkit.h"
int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_ERROR("Usage: ./quantized.out src.mnn dst.mnn preTreatConfig.json\n");
        return 0;
    }
    const char* modelFile      = argv[1];
    const char* preTreatConfig = argv[3];
    const char* dstFile        = argv[2];
    FUNC_PRINT_ALL(modelFile, s);
    FUNC_PRINT_ALL(preTreatConfig, s);
    FUNC_PRINT_ALL(dstFile, s);
    std::unique_ptr<MNN::NetT> netT;
    {
        std::ifstream input(modelFile);
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
    DLOG(INFO) << "calibrate the feature according to KL-divergence and quantize model...";
    std::shared_ptr<Calibration> calibration(
        new Calibration(netT.get(), modelForInference.get(), size, preTreatConfig));
    calibration->runQuantizeModel();
    DLOG(INFO) << "quantize model done!";

    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, netT.get());
    builderOutput.Finish(len);

    {
        std::ofstream output(dstFile);
        output.write((const char*)builderOutput.GetBufferPointer(), builderOutput.GetSize());
    }
}
