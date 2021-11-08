//
//  Program.hpp
//  MNNConverter
//
//  Created by MNN on 2019/09/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Program_hpp
#define Program_hpp
#include <fstream>
#include <map>
#include <unordered_map>
#include <sstream>
#include <string>
#include <set>
#include <MNN/expr/Expr.hpp>
namespace MNN {
namespace Express {

class Program {
public:
    static std::shared_ptr<Program> create(const MNN::NetT* net, bool supportExtra, bool saveAllVars = false);
    std::vector<VARP> outputs() const {
        return mOutputs;
    }
    void input(const std::unordered_map<std::string, VARP>& inputs);
    static void createUnit(std::map<int, VARP>& varMap, std::vector<int>& inputIndexes, const std::vector<std::unique_ptr<OpT>>& oplists, MNN::OpT* op, const MNN::NetT* net, std::set<OpT*>& invalidSet, std::set<int>& extraInputIndexes);

    const std::map<int, VARP>& vars() const {
        return mVars;
    }
private:
    Program() {
    }
    std::map<int, VARP> mVars;
    std::vector<VARP> mOutputs;
};
} // namespace Express
}; // namespace MNN

#endif
