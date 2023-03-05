//
//  OpenCLTarget.hpp
//  MNN
//
//  Created by MNN on 2022/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "../SourceModule.hpp"
namespace MNN {

class OpenCLTarget : public Target {
public:
    OpenCLTarget() {}
    ~OpenCLTarget() {}
    std::string codegen(std::vector<std::string>& inputs, const MNN::Op* op) override;
private:
    std::string type() override;
    std::string macro() override;
    std::string number(float val) override;
    std::string load(const std::string& base, const std::string& offset) override;
    std::string loadscalar(const std::string& base) override;
    std::string store(const std::string base, const std::string& offset, const std::string& data) override;
    std::string proto(const std::string& name, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs) override;
    template <typename T>
    std::string numval(T t) { return "((float4)" + std::to_string(t) + ")"; }
};

}
