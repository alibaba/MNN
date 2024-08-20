#include <cmath>
#include <fstream>
#include <sstream>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#define DUMP_NUM_DATA(type)                          \
    auto data = tensor->host<type>();                \
    for (int z = 0; z < outside; ++z) {              \
        for (int x = 0; x < width; ++x) {            \
            outputOs << data[x + z * width] << "\t"; \
        }                                            \
        outputOs << "\n";                            \
    }

#define DUMP_CHAR_DATA(type)                                           \
    auto data = tensor->host<type>();                                  \
    for (int z = 0; z < outside; ++z) {                                \
        for (int x = 0; x < width; ++x) {                              \
            outputOs << static_cast<int>(data[x + z * width]) << "\t"; \
        }                                                              \
        outputOs << "\n";                                              \
    }

static void dumpTensor2File(const MNN::Tensor* tensor, const char* file, std::ofstream& orderFile) {
    orderFile << file << std::endl;
    std::ofstream outputOs(file);
    auto type = tensor->getType();

    int dimension = tensor->buffer().dimensions;
    int width     = 1;
    if (dimension > 1) {
        width = tensor->length(dimension - 1);
    }

    const int outside = tensor->elementSize() / width;

    const auto dataType  = type.code;
    const auto dataBytes = type.bytes();

    if (dataType == halide_type_float) {
        DUMP_NUM_DATA(float);
    }
    if (dataType == halide_type_int && dataBytes == 4) {
        DUMP_NUM_DATA(int32_t);
    }
    if (dataType == halide_type_uint && dataBytes == 1) {
        DUMP_CHAR_DATA(uint8_t);
    }
    if (dataType == halide_type_int && dataBytes == 1) {
#ifdef MNN_USE_SSE
        auto data = tensor->host<uint8_t>();
        for (int z = 0; z < outside; ++z) {
            for (int x = 0; x < width; ++x) {
                outputOs << (static_cast<int>(data[x + z * width]) - 128) << "\t";
            }
            outputOs << "\n";
        }
#else
        DUMP_CHAR_DATA(int8_t);
#endif
    }
}

std::ofstream gOrderFile;
static void _initDebug() {
    gOrderFile.open("order.txt");
    MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const MNN::OperatorInfo* info) {
        auto opName = info->name();
        if (info->type() == "Copy") {
            return true;
        }
        auto opCopyName = opName;
        for (int j = 0; j < opCopyName.size(); ++j) {
            if (opCopyName[j] == '/') {
                opCopyName[j] = '_';
            }
        }
        for (int i = 0; i < ntensors.size(); ++i) {
            auto ntensor    = ntensors[i];
            auto outDimType = ntensor->getDimensionType();
            std::shared_ptr<MNN::Tensor> expectTensor(new MNN::Tensor(ntensor, outDimType));
            bool res = ntensor->copyToHostTensor(expectTensor.get());
            if (res) {
                ntensor = expectTensor.get();
            }
            std::ostringstream outputFileName;
            outputFileName << "output/Input_" << opCopyName << "_" << i;
            dumpTensor2File(ntensor, outputFileName.str().c_str(), gOrderFile);
        }
        return true;
    };
    MNN::TensorCallBackWithInfo callBack = [&](const std::vector<MNN::Tensor*>& ntensors,  const MNN::OperatorInfo* info) {
        auto opName = info->name();
        if (info->type() == "Copy") {
            return true;
        }
        for (int i = 0; i < ntensors.size(); ++i) {
            auto ntensor    = ntensors[i];
            auto outDimType = ntensor->getDimensionType();
            std::shared_ptr<MNN::Tensor> expectTensor(new MNN::Tensor(ntensor, outDimType));
            bool res = ntensor->copyToHostTensor(expectTensor.get());
            if (res) {
                ntensor = expectTensor.get();
            }
            std::ostringstream outputFileName;
            auto opCopyName = opName;
            for (int j = 0; j < opCopyName.size(); ++j) {
                if (opCopyName[j] == '/') {
                    opCopyName[j] = '_';
                }
            }
            auto tensor = ntensor;
            if (tensor->dimensions() == 4) {
                MNN_PRINT("Dimensions: 4, W,H,C,B: %d X %d X %d X %d, OP name %s : %d\n",
                        tensor->width(), tensor->height(), tensor->channel(), tensor->batch(), opName.c_str(), i);
            } else {
                std::ostringstream oss;
                for (int i = 0; i < tensor->dimensions(); i++) {
                    oss << (i ? " X " : "") << tensor->length(i);
                }

                MNN_PRINT("Dimensions: %d, %s, OP name %s : %d\n", tensor->dimensions(), oss.str().c_str(), opName.c_str(), i);
            }

            outputFileName << "output/" << opCopyName << "_" << i;
            dumpTensor2File(tensor, outputFileName.str().c_str(), gOrderFile);
        }
        return true;
    };
    MNN::Express::ExecutorScope::Current()->setCallBack(std::move(beforeCallBack), std::move(callBack));
}


struct TimeTraceInfo {
    std::map<std::string, std::map<std::string, std::vector<std::pair<float, float>>>> mTypes;
    
    void begin(const MNN::OperatorInfo* info) {
        auto tIter = mTypes.find(info->type());
        if (tIter == mTypes.end()) {
            std::map<std::string, std::vector<std::pair<float, float>>> _t;
            mTypes.insert(std::make_pair(info->type(), _t));
            tIter = mTypes.find(info->type());
        }
        mInserIter = tIter->second.find(info->name());
        if (mInserIter == tIter->second.end()) {
            std::vector<std::pair<float, float>> _t;
            tIter->second.insert(std::make_pair(info->name(), _t));
            mInserIter = tIter->second.find(info->name());
        }
        mTimer.reset();
    }
    void end(const MNN::OperatorInfo* info) {
        auto timeInMs = (float)mTimer.durationInUs() / 1000.0f;
        mInserIter->second.emplace_back(std::make_pair(timeInMs, info->flops()));
    }
private:
    std::map<std::string, std::vector<std::pair<float, float>>>::iterator mInserIter;
    MNN::Timer mTimer;
};
static TimeTraceInfo* gTimeTraceInfo = nullptr;
static void _initTimeTrace() {
    static TimeTraceInfo gTime;
    gTimeTraceInfo = &gTime;
    MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const MNN::OperatorInfo* info) {
        gTimeTraceInfo->begin(info);
        return true;
    };
    MNN::TensorCallBackWithInfo callBack = [&](const std::vector<MNN::Tensor*>& ntensors,  const MNN::OperatorInfo* info) {
        for (auto t : ntensors) {
            t->wait(MNN::Tensor::MAP_TENSOR_READ, true);
        }
        gTimeTraceInfo->end(info);
        return true;
    };
    MNN::Express::ExecutorScope::Current()->setCallBack(std::move(beforeCallBack), std::move(callBack));
}

template<typename T>
std::tuple<float, float, float> _countTensor(MNN::Tensor* tensor) {
    auto size = tensor->elementSize();
    auto ptr =  (T*)tensor->buffer().host;
    float maxValue = (float)ptr[0];
    float avgValue = (float)ptr[0];
    float minValue = (float)ptr[0];
    float sumDiv = 1.0f / (float)size;
    for (int i=1; i<size; ++i) {
        maxValue = fmaxf(maxValue, (float)ptr[i]);
        minValue = fminf(minValue, (float)ptr[i]);
        avgValue += (float)ptr[i] * sumDiv;
    }
    return std::make_tuple(maxValue, minValue, avgValue);
}

std::pair<bool, std::tuple<float, float, float>> _countForTensorValid(MNN::Tensor* ntensor) {
    bool valid = false;
    std::tuple<float, float, float> res;
    if (ntensor->elementSize() <= 0) {
        return std::make_pair(valid, res);
    }
    bool validforType = false;
    if (ntensor->getType().code == halide_type_float || ntensor->getType().code == halide_type_int || ntensor->getType().code == halide_type_uint) {
        validforType = true;
    }
    if (!validforType) {
        return std::make_pair(valid, res);
    }
    valid = true;
    auto outDimType = ntensor->getDimensionType();
    std::shared_ptr<MNN::Tensor> expectTensor(new MNN::Tensor(ntensor, outDimType));
    bool copyRes = ntensor->copyToHostTensor(expectTensor.get());
    if (copyRes) {
        ntensor = expectTensor.get();
    }
    std::tuple<float, float, float> data;
    if (ntensor->getType().code == halide_type_float) {
        data = _countTensor<float>(ntensor);
    } else if (ntensor->getType().code == halide_type_int) {
        if (ntensor->getType().bits == 32) {
            data = _countTensor<int32_t>(ntensor);
        } else if (ntensor->getType().bits == 8) {
            data = _countTensor<int8_t>(ntensor);
        }
    } else if (ntensor->getType().code == halide_type_uint) {
        if (ntensor->getType().bits == 32) {
            data = _countTensor<uint32_t>(ntensor);
        } else if (ntensor->getType().bits == 8) {
            data = _countTensor<uint8_t>(ntensor);
        }
    }
    return std::make_pair(valid, data);
}
static void _initTensorStatic() {
    MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const MNN::OperatorInfo* info) {
        auto opName = info->name();
        if (info->type() == "Copy") {
            return true;
        }
        for (int i = 0; i < ntensors.size(); ++i) {
            auto ntensor    = ntensors[i];
            auto res = _countForTensorValid(ntensor);
            if (!res.first) {
                continue;
            }
            auto data = res.second;
            MNN_PRINT("%s [Input] %s_%d, type:%d-%d, Max: %f, Min: %f, Avg: %f, [", info->type().c_str(), opName.c_str(), i,  ntensor->getType().code, ntensor->getType().bits, std::get<0>(data), std::get<1>(data), std::get<2>(data));
            for (int v=0; v<ntensor->dimensions(); ++v) {
                MNN_PRINT("%d", ntensor->length(v));
                if (v!=ntensor->dimensions()-1) {
                    MNN_PRINT(",");
                }
            }
            MNN_PRINT("]\n");
        }
        return true;
    };
    MNN::TensorCallBackWithInfo callBack = [&](const std::vector<MNN::Tensor*>& ntensors,  const MNN::OperatorInfo* info) {
        auto opName = info->name();
        if (info->type() == "Copy") {
            return true;
        }
        for (int i = 0; i < ntensors.size(); ++i) {
            auto ntensor    = ntensors[i];
            auto res = _countForTensorValid(ntensor);
            if (!res.first) {
                continue;
            }
            auto data = res.second;
            MNN_PRINT("%s [Output] %s_%d, type:%d-%d, Max: %f, Min: %f, Avg: %f, [", info->type().c_str(), opName.c_str(), i,  ntensor->getType().code, ntensor->getType().bits, std::get<0>(data), std::get<1>(data), std::get<2>(data));
            for (int v=0; v<ntensor->dimensions(); ++v) {
                MNN_PRINT("%d", ntensor->length(v));
                if (v!=ntensor->dimensions()-1) {
                    MNN_PRINT(",");
                }
            }
            MNN_PRINT("]\n");
        }
        return true;
    };
    MNN::Express::ExecutorScope::Current()->setCallBack(std::move(beforeCallBack), std::move(callBack));
}
