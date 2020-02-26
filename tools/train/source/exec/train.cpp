//
//  train.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include <math.h>
#include <stdlib.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <algorithm>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include "MNN_generated.h"
#include "core/Macro.h"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace MNN;
using namespace std;
std::random_device gDevice;
inline std::string numberToString(int index) {
    std::ostringstream os;
    os << index;
    return os.str();
}
template<typename T>
inline T stringConvert(const char* number) {
    std::istringstream os(number);
    T v;
    os >> v;
    return v;
}
static void dumpTensorToFile(const Tensor* tensor, std::string fileName) {
    std::unique_ptr<Tensor> hostTensor(new Tensor(tensor, MNN::Tensor::TENSORFLOW, true));
    tensor->copyToHostTensor(hostTensor.get());
    if (tensor->getType().code == halide_type_float) {
        auto origin0 = hostTensor->host<float>();
        std::ofstream prob(fileName);
        auto size = hostTensor->elementSize();
        for (int i = 0; i < size; ++i) {
            prob << origin0[i] << "\n";
        }
    } else if (tensor->getType().code == halide_type_int && tensor->getType().bytes() == 4) {
        auto origin0 = hostTensor->host<int32_t>();
        std::ofstream prob(fileName);
        auto size = hostTensor->elementSize();
        for (int i = 0; i < size; ++i) {
            prob << origin0[i] << "\n";
        }
    }
}
#define TEST_TRAIN
int main(int argc, const char* argv[]) {
    if (argc < 5) {
        MNN_PRINT(
            "Usage: ./train.out model.mnn data.bin test.bin times [learningRate] [LossName] [backend "
            "{0:CPU,1:OPENCL}]\n");
        return 0;
    }
    unique_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));

    int time      = atoi(argv[4]);
    int trainStep = 1;
    float lr      = 0.00001f;
    if (argc > 5) {
        lr = stringConvert<float>(argv[5]);
    }
    std::string lossName = "Loss";
    if (argc > 6) {
        lossName = argv[6];
    }
    ScheduleConfig config;
    if (argc > 7) {
        int backend = stringConvert<int>(argv[7]);
        if (backend == 1) {
            config.type = MNN_FORWARD_OPENCL;
        }
    }
    config.numThread = 1;
    config.saveTensors.emplace_back(lossName);
    BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    config.backendConfig    = &backendConfig;
    auto session            = net->createSession(config);
    auto loss               = net->getSessionOutput(session, lossName.c_str());
    std::unique_ptr<Tensor> lossHost(Tensor::createHostTensorFromDevice(loss, false));
    int maxBatch = 0;
    if (nullptr == loss) {
        MNN_ERROR("Can't find loss\n");
        return 0;
    }
    int batch = 1;

    std::map<std::string, std::tuple<std::unique_ptr<Tensor>, std::unique_ptr<Tensor>, Tensor*>> tensorInputStorage;
    {
        unique_ptr<Interpreter> dataNet(Interpreter::createFromFile(argv[2]));
        auto buffer = dataNet->getModelBuffer();
        auto netC   = GetNet(buffer.first);
        for (int i = 0; i < netC->oplists()->size(); ++i) {
            MNN_ASSERT(OpType_Const == netC->oplists()->GetAs<Op>(i)->type());
            auto blob = netC->oplists()->GetAs<Op>(i)->main_as_Blob();
            std::vector<int> dims;
            for (int d = 0; d < blob->dims()->size(); ++d) {
                dims.emplace_back(blob->dims()->data()[d]);
            }
            maxBatch = dims[0];
            FUNC_PRINT(maxBatch);
            auto dimFormat = blob->dataFormat();
            auto dimType   = Tensor::CAFFE;
            if (dimFormat == MNN_DATA_FORMAT_NHWC) {
                dimType = Tensor::TENSORFLOW;
            }
            void* sourcePtr = nullptr;
            switch (blob->dataType()) {
                case MNN::DataType_DT_FLOAT:
                    sourcePtr = (void*)(blob->float32s()->data());
                    break;
                case MNN::DataType_DT_INT32:
                    sourcePtr = (void*)(blob->int32s()->data());
                    break;

                default:
                    break;
            }
            unique_ptr<Tensor> tensor(Tensor::create<float>(dims, nullptr, dimType));
            ::memcpy(tensor->host<float>(), sourcePtr, tensor->size());
            auto name        = netC->oplists()->GetAs<Op>(i)->name()->str();
            auto inputOrigin = net->getSessionInput(session, name.c_str());
            batch            = inputOrigin->shape()[0];
            FUNC_PRINT(batch);
            std::unique_ptr<Tensor> inputOriginUser(new Tensor(inputOrigin, dimType));
            tensorInputStorage.insert(
                std::make_pair(name, std::make_tuple(std::move(tensor), std::move(inputOriginUser), inputOrigin)));

            FUNC_PRINT_ALL(name.c_str(), s);
        }
    }
    auto learnRate = net->getSessionInput(session, "LearningRate");
    std::unique_ptr<Tensor> learnRateHost(Tensor::createHostTensorFromDevice(learnRate, false));
    learnRateHost->host<float>()[0] = lr;
    TensorCallBack begin            = [](const std::vector<Tensor*>& inputs, const std::string& name) { return true; };
    TensorCallBack afterEval        = [lossName](const std::vector<Tensor*>& output, const std::string& name) {
        if (name == lossName) {
            return false;
        }
        return true;
    };

    int offset = 0;

    for (int l = 0; l < time; ++l) {
        AUTOTIME;
#ifndef TEST_TRAIN
        if (l % trainStep == 0) {
            float meanloss = 0.0f;
            // Eval Train Loss
            int batchSize = maxBatch / batch;
            for (int v = 0; v < batchSize; ++v) {
                for (int n = 0; n < batch; ++n) {
                    int index = n + batch * v;
                    for (auto& iter : tensorInputStorage) {
                        auto& src = get<0>(iter.second);
                        auto& dst = get<1>(iter.second);
                        ::memcpy(dst->host<float>() + n * dst->stride(0), src->host<float>() + index * src->stride(0),
                                 src->stride(0) * sizeof(float));
                    }
                }
                for (auto& iter : tensorInputStorage) {
                    auto& src = get<1>(iter.second);
                    auto& dst = get<2>(iter.second);
                    dst->copyFromHostTensor(src.get());
                }
                learnRate->copyFromHostTensor(learnRateHost.get());
                net->runSessionWithCallBack(session, begin, afterEval, true);
                loss->copyToHostTensor(lossHost.get());
                meanloss += lossHost->host<float>()[0];
            }
            meanloss = meanloss / ((float)batchSize * batch);
            FUNC_PRINT_ALL(meanloss, f);
        }
#endif
        for (int n = 0; n < batch; ++n) {
#ifndef TEST_TRAIN
            int index = gDevice() % maxBatch;
#else
            int index = offset + n;
#endif
            for (auto& iter : tensorInputStorage) {
                auto& src = get<0>(iter.second);
                auto& dst = get<1>(iter.second);
                ::memcpy(dst->host<float>() + n * dst->stride(0), src->host<float>() + index * src->stride(0),
                         src->stride(0) * sizeof(float));
            }
        }
        for (auto& iter : tensorInputStorage) {
            auto& src = get<1>(iter.second);
            auto& dst = get<2>(iter.second);
            dst->copyFromHostTensor(src.get());
        }
        learnRate->copyFromHostTensor(learnRateHost.get());
        net->runSession(session);
#ifdef TEST_TRAIN
        static float historyLossValue = 1000000.0f;
        loss->copyToHostTensor(lossHost.get());
        auto lossValue = lossHost->host<float>()[0];
        FUNC_PRINT_ALL(lossValue, f);
        if (lossValue > historyLossValue) {
            MNN_ERROR("Loss value error, from %f to %f \n", historyLossValue, lossValue);
            break;
        }
        historyLossValue = lossValue;
#endif
    }

    if (true) {
        TensorCallBack begin = [](const std::vector<Tensor*>& inputs, const std::string& oname) {
            std::string name = oname;
            for (int i = 0; i < name.size(); ++i) {
                if (name[i] == '/') {
                    name[i] = '_';
                }
            }
            for (int index = 0; index < inputs.size(); ++index) {
                auto fileName = std::string("output/") + name + "_input_" + numberToString(index);
                dumpTensorToFile(inputs[index], fileName);
            }
            return true;
        };
        TensorCallBack after = [lossName](const std::vector<Tensor*>& output, const std::string& oname) {
            std::string name = oname;
            for (int i = 0; i < name.size(); ++i) {
                if (name[i] == '/') {
                    name[i] = '_';
                }
            }
            float maxValue = 0.0f;
            std::unique_ptr<Tensor> hostOutput;
            for (int index = 0; index < output.size(); ++index) {
                if (output[index]->getType().code != halide_type_float) {
                    continue;
                }
                std::ofstream prob("output/" + name + "_" + numberToString(index));
                hostOutput.reset(new Tensor(output[index], MNN::Tensor::TENSORFLOW, true));
                output[index]->copyToHostTensor(hostOutput.get());
                auto origin0 = hostOutput->host<float>();
                auto size    = hostOutput->elementSize();
                for (int i = 0; i < size; ++i) {
                    auto value = origin0[i];
                    if ((!(value > 0.0f)) && (!(value <= 0.0f))) {
                        maxValue = 12345; // NAN
                        break;
                    }
                    maxValue = std::max(maxValue, fabsf(origin0[i]));
                    prob << origin0[i] << "\n";
                }
            }
            if (maxValue > 10000.0f) {
                MNN_PRINT("Invalid value : %f, %s\n", maxValue, oname.c_str());
            }
            return true;
        };
        for (int n = 0; n < batch; ++n) {
            int index = offset + n;
            for (auto& iter : tensorInputStorage) {
                auto& src = get<0>(iter.second);
                auto& dst = get<1>(iter.second);
                ::memcpy(dst->host<float>() + n * dst->stride(0), src->host<float>() + index * src->stride(0),
                         src->stride(0) * sizeof(float));
            }
        }
        for (auto& iter : tensorInputStorage) {
            auto& src = get<1>(iter.second);
            auto& dst = get<2>(iter.second);
            dst->copyFromHostTensor(src.get());
            //            auto fileName = iter.first;
            //            for (int i = 0; i < fileName.size(); ++i) {
            //                if (fileName[i] == '/') {
            //                    fileName[i] = '_';
            //                }
            //            }
            //            dumpTensorToFile(src.get(), "output/Input_Src_" + fileName);
            //            dumpTensorToFile(dst, "output/Input_Dst_" + fileName);
        }
        learnRate->copyFromHostTensor(learnRateHost.get());
        net->runSessionWithCallBack(session, begin, after, true);
    }
    net->updateSessionToModel(session);
    {
        auto modelBuffer = net->getModelBuffer();
        ofstream output("trainResult.bin");
        output.write((const char*)modelBuffer.first, modelBuffer.second);
    }

    return 0;
}
