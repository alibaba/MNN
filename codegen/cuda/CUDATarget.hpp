//
//  CUDATarget.hpp
//  MNN
//
//  Created by MNN on 2023/06/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "../SourceModule.hpp"
namespace MNN {

class CUDATarget : public Target {
public:
    CUDATarget(const BackendConfig::PrecisionMode precision) : Target(precision) {}
    ~CUDATarget() {}
    std::string codegen(std::vector<std::string>& inputs, const Command* cmd, std::string& inpName) override;
private:
    std::string type() override;
    std::string macro() override;
    std::string number(float val) override;
    std::string load(const std::string& base, const std::string& offset, const Command* cmd, std::string& inpName) override;
    std::string loadscalar(const std::string& base, std::string& inpName) override;
    std::string store(const std::string base, const std::string& offset, const std::string& data) override;
    std::string proto(const std::string& name, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, bool hasSingleConvertRaster = false) override;
    template <typename T>
    std::string numval(T t) { return "(float)(" + std::to_string(t) + ")"; }
private:
    OpType mOpType;
};

}
