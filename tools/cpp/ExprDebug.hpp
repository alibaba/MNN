#include <cmath>
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
    MNN::Express::Executor::getGlobalExecutor()->setCallBack(std::move(beforeCallBack), std::move(callBack));
}


struct TimeTraceInfo {
    std::map<std::string, std::map<std::string, std::vector<float>>> mTypes;
    
    void begin(const MNN::OperatorInfo* info) {
        auto tIter = mTypes.find(info->type());
        if (tIter == mTypes.end()) {
            std::map<std::string, std::vector<float>> _t;
            mTypes.insert(std::make_pair(info->type(), _t));
            tIter = mTypes.find(info->type());
        }
        mInserIter = tIter->second.find(info->name());
        if (mInserIter == tIter->second.end()) {
            std::vector<float> _t;
            tIter->second.insert(std::make_pair(info->name(), _t));
            mInserIter = tIter->second.find(info->name());
        }
        mTimer.reset();
    }
    void end() {
        auto timeInMs = (float)mTimer.durationInUs() / 1000.0f;
        mInserIter->second.emplace_back(timeInMs);
    }
private:
    std::map<std::string, std::vector<float>>::iterator mInserIter;
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
        gTimeTraceInfo->end();
        return true;
    };
    MNN::Express::Executor::getGlobalExecutor()->setCallBack(std::move(beforeCallBack), std::move(callBack));
}
static std::tuple<float, float, float> _countTensor(MNN::Tensor* tensor) {
    auto size = tensor->elementSize();
    auto ptr =  tensor->host<float>();
    float maxValue = ptr[0];
    float sumValue = ptr[0];
    float minValue = ptr[0];
    for (int i=1; i<size; ++i) {
        maxValue = fmaxf(maxValue, ptr[i]);
        minValue = fminf(minValue, ptr[i]);
        sumValue += ptr[i];
    }
    auto avgValue = sumValue / (float)size;
    return std::make_tuple(maxValue, minValue, avgValue);
}
static void _initTensorStatic() {
    MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const MNN::OperatorInfo* info) {
        auto opName = info->name();
        if (info->type() == "Copy") {
            return true;
        }
        for (int i = 0; i < ntensors.size(); ++i) {
            auto ntensor    = ntensors[i];
            if (ntensor->getType().code != halide_type_float || ntensor->elementSize() <= 0) {
                continue;
            }
            auto outDimType = ntensor->getDimensionType();
            std::shared_ptr<MNN::Tensor> expectTensor(new MNN::Tensor(ntensor, outDimType));
            bool res = ntensor->copyToHostTensor(expectTensor.get());
            if (res) {
                ntensor = expectTensor.get();
            }
            auto data = _countTensor(ntensor);
            MNN_PRINT("%s [Input] %s_%d, Max: %f, Min: %f, Avg: %f, [", info->type().c_str(), opName.c_str(), i, std::get<0>(data), std::get<1>(data), std::get<2>(data));
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
            if (ntensor->getType().code != halide_type_float || ntensor->elementSize() <= 0) {
                continue;
            }
            auto outDimType = ntensor->getDimensionType();
            std::shared_ptr<MNN::Tensor> expectTensor(new MNN::Tensor(ntensor, outDimType));
            bool res = ntensor->copyToHostTensor(expectTensor.get());
            if (res) {
                ntensor = expectTensor.get();
            }
            auto data = _countTensor(ntensor);
            MNN_PRINT("%s [Output] %s_%d, Max: %f, Min: %f, Avg: %f, [", info->type().c_str(), opName.c_str(), i, std::get<0>(data), std::get<1>(data), std::get<2>(data));
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
    MNN::Express::Executor::getGlobalExecutor()->setCallBack(std::move(beforeCallBack), std::move(callBack));
}
