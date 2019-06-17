//
//  train.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "MNN_generated.h"
#include "Macro.h"
#include "Tensor.hpp"
//#define MNN_OPEN_TIME_TRACE
#include "AutoTime.hpp"
using namespace MNN;
using namespace std;
std::random_device gDevice;
inline std::string numberToString(int index) {
    std::ostringstream os;
    os << index;
    return os.str();
}

//#define TEST_TRAIN
int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./train.out temp.bin data.bin times learningRate batch [LossName]\n");
        return 0;
    }
    unique_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));

    int time  = atoi(argv[3]);
    float lr  = -0.00001f;
    int batch = 1;
    if (argc > 5) {
        batch = atoi(argv[5]);
    }
    if (argc > 4) {
        lr = -atof(argv[4]) / (float)batch;
    }
    std::string lossName = "Loss";
    if (argc > 6) {
        lossName = argv[6];
    }
    ScheduleConfig config;
    config.numThread = 4;
    config.saveTensors.emplace_back(lossName);
    auto session = net->createSession(config);
    auto loss    = net->getSessionOutput(session, lossName.c_str());
    int maxBatch = 0;
    if (nullptr == loss) {
        MNN_ERROR("Can't find loss\n");
        return 0;
    }

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
            auto shape       = dims;
            shape[0]         = batch;
            net->resizeTensor(inputOrigin, shape);
            if (inputOrigin->elementSize() <= 0) {
                MNN_ERROR("Error: batch = %d\n", batch);
                return 0;
            }
            std::unique_ptr<Tensor> inputOriginUser(new Tensor(inputOrigin));
            tensorInputStorage.insert(
                std::make_pair(name, std::make_tuple(std::move(tensor), std::move(inputOriginUser), inputOrigin)));

            FUNC_PRINT_ALL(name.c_str(), s);
        }
    }
    net->resizeSession(session);
    auto learnRate       = net->getSessionInput(session, "LearningRate");
    TensorCallBack begin = [](const std::vector<Tensor*>& inputs, const std::string& name) { return true; };
    TensorCallBack after = [](const std::vector<Tensor*>& output, const std::string& name) {
        //        if (name == "Loss_Grad") {
        //            ::memset(output[0]->host<float>(), 0, output[0]->size());
        //        }
        return true;
    };
    TensorCallBack afterEval = [lossName](const std::vector<Tensor*>& output, const std::string& name) {
        if (name == lossName) {
            return false;
        }
        return true;
    };

    int trainStep = 50;
    int offset    = 0;

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
                learnRate->host<float>()[0] = lr;
                net->runSessionWithCallBack(session, begin, afterEval);
                meanloss += loss->host<float>()[0];
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
        learnRate->host<float>()[0] = lr;
        net->runSessionWithCallBack(session, begin, after);
#ifdef TEST_TRAIN
        static float historyLossValue = 1000000.0f;
        auto lossValue                = loss->host<float>()[0];
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
                if (inputs[index]->getType().code != halide_type_float) {
                    continue;
                }
                auto origin0 = inputs[index]->host<float>();
                std::ofstream prob("output/" + name + "_input_" + numberToString(index));
                auto size = inputs[index]->elementSize();
                for (int i = 0; i < size; ++i) {
                    prob << origin0[i] << "\n";
                }
            }
            return true;
        };
        TensorCallBack after = [](const std::vector<Tensor*>& output, const std::string& oname) {
            std::string name = oname;
            for (int i = 0; i < name.size(); ++i) {
                if (name[i] == '/') {
                    name[i] = '_';
                }
            }
            float maxValue = 0.0f;
            for (int index = 0; index < output.size(); ++index) {
                if (output[index]->getType().code != halide_type_float) {
                    continue;
                }
                std::ofstream prob("output/" + name + "_" + numberToString(index));
                auto origin0 = output[index]->host<float>();
                auto size    = output[index]->elementSize();
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
        learnRate->host<float>()[0] = lr;
        net->runSessionWithCallBack(session, begin, after);
    }
    net->updateSessionToModel(session);
    {
        auto modelBuffer = net->getModelBuffer();
        ofstream output("trainResult.bin");
        output.write((const char*)modelBuffer.first, modelBuffer.second);
    }

    return 0;
}
