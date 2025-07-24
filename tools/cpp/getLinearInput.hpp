#ifndef GET_LINEAR_INPUT_HPP
#define GET_LINEAR_INPUT_HPP

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/Tensor.hpp>
#include <mutex>
#include <fstream>
#include <cstdlib>
#include <string>

namespace MNN {
namespace LinearInput {

static std::ofstream thresholdFile;
static std::mutex thresholdFileMutex;
static bool isFirstThreshold = true;
static std::string currentThresholdFile = "thresholds.json";

static void closeThresholdFile() {
    std::lock_guard<std::mutex> lock(thresholdFileMutex);
    
    if (thresholdFile.is_open()) {
        thresholdFile << "\n}\n";
        thresholdFile.close();
        MNN_PRINT("Threshold file closed: %s\n", currentThresholdFile.c_str());
    }
}

static void initThresholdFile(const std::string& filename = "thresholds.json") {
    std::lock_guard<std::mutex> lock(thresholdFileMutex);
    
    if (thresholdFile.is_open()) {
        thresholdFile.close();
    }
    
    currentThresholdFile = filename;
    thresholdFile.open(filename, std::ios::out);
    if (thresholdFile.is_open()) {
        thresholdFile << "{\n";
        thresholdFile.flush();
        isFirstThreshold = true;
        MNN_PRINT("Initialized threshold file: %s\n", filename.c_str());
    } else {
        MNN_ERROR("Failed to open threshold file: %s\n", filename.c_str());
    }
}

static void writeThresholdRealtime(const std::string& opName, float thresholdValue) {
    std::lock_guard<std::mutex> lock(thresholdFileMutex);
    
    if (!thresholdFile.is_open()) {
        initThresholdFile(currentThresholdFile);
    }
    
    if (thresholdFile.is_open()) {
        if (!isFirstThreshold) {
            thresholdFile << ",\n";
        }
        
        thresholdFile << "    \"" << opName << "\": " << thresholdValue;
        thresholdFile.flush();  
        isFirstThreshold = false;
        
        MNN_PRINT("Saved threshold: %s = %f\n", opName.c_str(), thresholdValue);
    }
}

static std::ofstream maxValueFile;
static std::mutex maxValueFileMutex;
static bool isFirstMaxValue = true;
static std::string currentMaxValueFile = "max_values.json";

static void closeMaxValueFile() {
    std::lock_guard<std::mutex> lock(maxValueFileMutex);
    
    if (maxValueFile.is_open()) {
        maxValueFile << "\n}\n";
        maxValueFile.close();
        MNN_PRINT("Max value file closed: %s\n", currentMaxValueFile.c_str());
    }
}

static void initMaxValueFile(const std::string& filename = "max_values.json") {
    std::lock_guard<std::mutex> lock(maxValueFileMutex);
    
    if (maxValueFile.is_open()) {
        maxValueFile.close();
    }
    
    currentMaxValueFile = filename;
    maxValueFile.open(filename, std::ios::out);
    if (maxValueFile.is_open()) {
        maxValueFile << "{\n";
        maxValueFile.flush();
        isFirstMaxValue = true;
        MNN_PRINT("Initialized max value file: %s\n", filename.c_str());
    } else {
        MNN_ERROR("Failed to open max value file: %s\n", filename.c_str());
    }
}

static void writeMaxValueRealtime(const std::string& opName, float maxValueFloat) {
    std::lock_guard<std::mutex> lock(maxValueFileMutex);
    
    if (!maxValueFile.is_open()) {
        initMaxValueFile(currentMaxValueFile);
    }
    
    if (maxValueFile.is_open()) {
        if (!isFirstMaxValue) {
            maxValueFile << ",\n";
        }
        
        maxValueFile << "    \"" << opName << "\": " << maxValueFloat;
        maxValueFile.flush();  
        isFirstMaxValue = false;
        
        MNN_PRINT("Saved max value: %s = %f\n", opName.c_str(), maxValueFloat);
    }
}

static void cleanupAtExit() {
    closeThresholdFile();
    closeMaxValueFile();
}


inline void initGetThreshold(const std::string& thresholdFileName, float targetSparsity) {
    initThresholdFile(thresholdFileName);
    
    static bool registered = false;
    if (!registered) {
        std::atexit(cleanupAtExit);
        registered = true;
    }

    MNN::TensorCallBackWithInfo beforeCallBack = [targetSparsity](const std::vector<MNN::Tensor*>& ntensors, const MNN::OperatorInfo* info) {
        auto opName = info->name();
        if (info->type() == "Copy") {
            return true;
        }
        if (opName.find("Linear") == std::string::npos || opName.find("raster") != std::string::npos) {
            return true;
        }
        for (int i = 0; i < ntensors.size(); ++i) {
            auto ntensor = ntensors[i];
            auto outDimType = ntensor->getDimensionType();
            std::shared_ptr<MNN::Tensor> expectTensor(new MNN::Tensor(ntensor, outDimType));
            bool res = ntensor->copyToHostTensor(expectTensor.get());
            if (res) {
                ntensor = expectTensor.get();
            }
            {
                auto ninput = MNN::Express::Variable::create(MNN::Express::Expr::create(ntensor));
                if (nullptr == ninput->getInfo()) {
                    MNN_ERROR("Alloc memory or compute size error\n");
                    return false;
                }
                ninput = MNN::Express::_Convert(ninput, MNN::Express::NHWC);
                ninput = MNN::Express::_Abs(ninput);
                ninput = MNN::Express::_Reshape(ninput, {-1});

                auto totalNum = ninput->getInfo()->dim[0];
                int keepNum = totalNum * (1 - targetSparsity);

                auto kv = MNN::Express::_TopKV2(ninput, MNN::Express::_Scalar<int>(keepNum));
                auto values = kv[0];

                auto threshold = MNN::Express::_Gather(values, MNN::Express::_Scalar<int>(keepNum - 1));
                auto thresholdValue = threshold->readMap<float>()[0];

                writeThresholdRealtime(opName, thresholdValue);
            }
        }
        return true;
    };

    MNN::TensorCallBackWithInfo callBack = [](const std::vector<MNN::Tensor*>& ntensors, const MNN::OperatorInfo* info) {
        return true;
    };

    MNN::Express::ExecutorScope::Current()->setCallBack(std::move(beforeCallBack), std::move(callBack));
}

inline void initGetMaxValue(const std::string& maxValueFileName) {
    initMaxValueFile(maxValueFileName);
    
    static bool registered = false;
    if (!registered) {
        std::atexit(cleanupAtExit);
        registered = true;
    }

    MNN::TensorCallBackWithInfo beforeCallBack = [](const std::vector<MNN::Tensor*>& ntensors, const MNN::OperatorInfo* info) {
        auto opName = info->name();
        if (info->type() == "Copy") {
            return true;
        }
        if (opName.find("Linear") == std::string::npos || opName.find("raster") != std::string::npos) {
            return true;
        }

        for (int i = 0; i < ntensors.size(); ++i) {
            auto ntensor = ntensors[i];
            auto outDimType = ntensor->getDimensionType();
            std::shared_ptr<MNN::Tensor> expectTensor(new MNN::Tensor(ntensor, outDimType));
            bool res = ntensor->copyToHostTensor(expectTensor.get());
            if (res) {
                ntensor = expectTensor.get();
            }
            {
                auto ninput = MNN::Express::Variable::create(MNN::Express::Expr::create(ntensor));
                if (nullptr == ninput->getInfo()) {
                    MNN_ERROR("Alloc memory or compute size error\n");
                    return false;
                }
                ninput = MNN::Express::_Convert(ninput, MNN::Express::NHWC);
                ninput = MNN::Express::_Abs(ninput);  
                ninput = MNN::Express::_Reshape(ninput, {-1}); 

                auto kv = MNN::Express::_TopKV2(ninput, MNN::Express::_Scalar<int>(1));     
                auto maxValues = kv[0];  
                auto maxValueFloat = maxValues->readMap<float>()[0];

                writeMaxValueRealtime(opName, maxValueFloat);
            }
        }
        return true;
    };

    MNN::TensorCallBackWithInfo callBack = [](const std::vector<MNN::Tensor*>& ntensors, const MNN::OperatorInfo* info) {
        return true;
    };

    MNN::Express::ExecutorScope::Current()->setCallBack(std::move(beforeCallBack), std::move(callBack));
}

inline void closeAllFiles() {
    closeThresholdFile();
    closeMaxValueFile();
}

} // namespace LinearInput
} // namespace MNN

#endif // GET_LINEAR_INPUT_HPP