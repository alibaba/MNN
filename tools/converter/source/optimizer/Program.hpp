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
struct SubGraphProtoT;
namespace Express {

class Program {
public:
    static std::shared_ptr<Program> create(const MNN::NetT* net, bool supportExtra, bool saveAllVars = false);
    static std::shared_ptr<Program> create(const MNN::SubGraphProtoT* subgraph, bool supportExtra, bool saveAllVars = false);
    std::vector<VARP> outputs() const {
        return mOutputs;
    }
    VARPS input(const std::unordered_map<std::string, VARP>& inputs, bool lazy = false);
    static void createUnit(std::map<int, VARP>& varMap, std::vector<int>& inputIndexes, const std::vector<std::unique_ptr<OpT>>& oplists, MNN::OpT* op, const MNN::NetT* net, std::set<OpT*>& invalidSet, std::set<int>& extraInputIndexes);

    const std::map<int, VARP>& vars() const {
        return mVars;
    }
    void save(MNN::NetT* net);
private:
    static std::shared_ptr<Program> create(const std::vector<std::unique_ptr<OpT>>& oplists, const std::vector<std::string>& tensorName, const std::vector<std::string>& outputName, bool supportExtra, bool saveAllVars);
    static void createUnit(std::map<int, VARP>& varMap, std::vector<int>& inputIndexes, const std::vector<std::unique_ptr<OpT>>& oplists, MNN::OpT* op, const std::vector<std::string>& tensorName, std::set<OpT*>& invalidSet, std::set<int>& extraInputIndexes);
    Program() {
    }
    std::map<int, VARP> mVars;
    std::vector<VARP> mOutputs;
    std::vector<std::tuple<VARP, EXPRP, int>> mOriginInputs;
};
} // namespace Express
}; // namespace MNN

#endif
