//
//  Program.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Program.hpp"
#include <MNN/expr/ExprCreator.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace MNN::Express;
using namespace MNN;
#define UP_DIV(x) (((x) + 3) / 4)
#include "MNN_generated.h"
namespace MNN {
namespace Express {

static bool _isControlOp(const OpT* op) {
    std::set<std::string> controlOps{"Merge", "Switch", "LoopCond", "Enter", "Exit", "NextIteration"};
    return op->type == OpType_Extra && controlOps.find(op->main.AsExtra()->type) != controlOps.end();
}

struct Frame {
    std::vector<MNN::OpT*> body;
    std::vector<std::shared_ptr<Frame>> children;
    std::string name;
    std::string whileName;
    Frame* parent = nullptr;
    void reorder() {
        std::vector<OpT*> enter;
        std::vector<OpT*> other;
        std::vector<OpT*> exit;
        for (int i=0; i<body.size(); ++i) {
            if (nullptr != body[i] && body[i]->main.AsExtra()->type == "Enter") {
                enter.emplace_back(body[i]);
            } else if(nullptr != body[i] && body[i]->main.AsExtra()->type == "Exit") {
                exit.emplace_back(body[i]);
            } else {
                other.emplace_back(body[i]);
            }
        }
        body.clear();
        for (auto e : enter) {
            body.emplace_back(e);
        }
        for (auto o : other) {
            body.emplace_back(o);
        }
        for (auto e : exit) {
            body.emplace_back(e);
        }
    }
    void emit(const std::map<int, VARP>& context, std::ostream& output) {
        reorder();
        auto getName = [&](int index) {
            if (context.find(index) != context.end()) {
                auto name = context.find(index)->second->name();
                if (name.empty()) {
                    return std::string("VARP(nullptr)");
                }
                return "varMap[\"" + name + "\"]";
            }
            std::ostringstream os;
            os << "v" << index;
            return os.str();
        };
        int iter          = 0;
        bool inLoop       = false;
        int loopCondIndex = -1;
        std::map<int, OpT*> enters;
        std::map<int, OpT*> merges;
        std::map<int, OpT*> switches;
        for (auto op : body) {
            if (nullptr == op) {
                children[iter]->emit(context, output);
                iter++;
                continue;
            }
            std::vector<int> currentOutputIndex{op->outputIndexes[0]};
            std::shared_ptr<void> __defer(nullptr, [&](void*) {
                for (auto v : currentOutputIndex) {
                    if (context.find(v) != context.end()) {
                        auto nextName = context.find(v)->second->name();
                        auto index    = v;
                        output << "varMap[\"" << nextName << "\"]->input(v" << index << ");\n";
                    }
                }
            });

            auto type = op->main.AsExtra()->type;
            if ("Enter" == type) {
                output << "auto v" << op->outputIndexes[0] << " = " << getName(op->inputIndexes[0]) << ";\n";
                enters[op->outputIndexes[0]] = op;
                continue;
            }
            if ("Merge" == type) {
                if (enters.find(op->inputIndexes[0]) != enters.end()) {
                    // In circle Merge
                    merges[op->inputIndexes[1]] = op;
                    output << "auto v" << op->outputIndexes[0] << " = v" << op->inputIndexes[0] << ";\n";
                } else {
                    output << "VARP v" << op->outputIndexes[0] <<";\n do \n {\n";
                    for (auto index : op->inputIndexes) {
                        output << "if (" << getName(index) << "->getInfo() != nullptr) {\n";
                        output << "v" << op->outputIndexes[0] << " = " << getName(index) << ";\nbreak;\n}\n";
                    }
                    output << "} while (false);\n";
                }
                continue;
            }
            if ("LoopCond" == type) {
                output << "auto v" << op->outputIndexes[0] << " = " << getName(op->inputIndexes[0]) << ";\n";
                output << "while(v" << op->outputIndexes[0] << "->readMap<int>()[0] > 0) {\n";
                loopCondIndex = op->outputIndexes[0];
                inLoop        = true;
                continue;
            }
            if ("Switch" == type) {
                if (op->inputIndexes[1] == loopCondIndex) {
                    output << "auto v" << op->outputIndexes[1] << " = " << getName(op->inputIndexes[0]) << ";\n";
                    currentOutputIndex[0]          = op->outputIndexes[1];
                    switches[op->outputIndexes[0]] = op;
                } else {
                    currentOutputIndex = op->outputIndexes;
                    output << "VARP v" << op->outputIndexes[0] <<";\n";
                    if (currentOutputIndex.size() > 1) {
                        output << "VARP v" << op->outputIndexes[1] <<";\n";
                    }
                    output << "if (" << getName(op->inputIndexes[1]) << "->readMap<int>()[0] <= 0){\n";
                    output << "v" << op->outputIndexes[0] << " = " <<getName(op->inputIndexes[0]) <<";\n";
                    output << "}\n";
                    if (currentOutputIndex.size() > 1) {
                        output << "else {\n";
                        output << "v" << op->outputIndexes[1] << " = " <<getName(op->inputIndexes[0]) <<";\n";
                        output << "}\n";
                    }
                }
                continue;
            }
            if ("NextIteration" == type) {
                auto merge = merges.find(op->outputIndexes[0]);
                MNN_ASSERT(merge != merges.end());
                output << "v" << merge->second->outputIndexes[0] << " = _Clone(" << getName(op->inputIndexes[0]) << ", true);\n";
                currentOutputIndex[0] = merge->second->outputIndexes[0];
                continue;
            }
            if ("Exit" == type) {
                if (inLoop) {
                    inLoop = false;
                    output << "}\n";
                }
                auto switchIter = switches.find(op->inputIndexes[0]);
                MNN_ASSERT(switchIter != switches.end());
                output << "auto v" << op->outputIndexes[0] << " = v" << switchIter->second->inputIndexes[0] << ";\n";
                continue;
            }
            MNN_ASSERT(1 == op->outputIndexes.size() && 1 <= op->inputIndexes.size());
            output << "auto v" << op->outputIndexes[0] << " = " << op->main.AsExtra()->engine << "_"
                   << op->main.AsExtra()->type << "({";
            for (int v = 0; v < op->inputIndexes.size() - 1; ++v) {
                output << getName(op->inputIndexes[v]) << ", ";
            }
            output << getName(op->inputIndexes[op->inputIndexes.size() - 1]) << "});\n";
        }
    }

    void emitUtils(std::set<std::string>& emitted, std::ostream& output) {
        int iter = 0;
        for (auto op : body) {
            if (nullptr == op) {
                children[iter]->emitUtils(emitted, output);
                iter++;
                continue;
            }
            if (!_isControlOp(op) && OpType_Extra == op->type) {
                auto key = op->main.AsExtra()->engine + "_" + op->main.AsExtra()->type;
                if (emitted.find(key) == emitted.end()) {
                    output << "VARP " << key << "(std::vector<VARP> inputs) {\n";
                    output << "// Fill Content\n";
                    output << "}\n";
                    emitted.insert(key);
                }
            }
        }
    }
};
void Program::emit(std::ostream& output) {
    for (auto f : mFrames) {
        f->emit(mVars, output);
    }
}
void Program::emitUtils(std::ostream& output) {
    std::set<std::string> emitted;
    for (auto f : mFrames) {
        f->emitUtils(emitted, output);
    }
}
bool Program::needGenerateCode() const {
    return !mFrames.empty();
}

static void _create(std::map<int, VARP>& varMap, std::vector<int>& inputIndexes, const std::vector<OpT*>& oplists, int index, const MNN::NetT* net, std::set<OpT*>& invalidSet) {
    auto op = oplists[index];
    if (invalidSet.find(op) != invalidSet.end()) {
        return;
    }
    std::vector<VARP> inputVars;
    auto outputIndexes = op->outputIndexes;
    for (int j=0; j<outputIndexes.size(); ++j) {
        if (varMap.find(outputIndexes[j]) != varMap.end()) {
            // Don't support multi op output to one index
            return;
        }
    }
    invalidSet.insert(op);
    for (auto input : op->inputIndexes) {
        if (varMap.find(input) == varMap.end()) {
            for (int j = 0; j<oplists.size(); ++j) {
                for (auto outputIndex : oplists[j]->outputIndexes) {
                    if (outputIndex == input) {
                        _create(varMap, inputIndexes, oplists, j, net, invalidSet);
                    }
                }
            }
            if (varMap.find(input) == varMap.end()) {
                auto newInput = _Input();
                newInput->setName(net->tensorName[input]);
                varMap[input] = newInput;
            }
        }
        inputVars.emplace_back(varMap[input]);
    }
    auto expr          = Expr::create(op, inputVars, outputIndexes.size());
    expr->setName(op->name);
    for (int j = 0; j < outputIndexes.size(); ++j) {
        if (op->type == OpType_Input) {
            inputIndexes.emplace_back(outputIndexes[j]);
        }
        auto newVar = Variable::create(expr, j);
        newVar->setName(net->tensorName[outputIndexes[j]]);
        varMap[outputIndexes[j]] = newVar;
    }
}

std::shared_ptr<Program> Program::create(const MNN::NetT* net, bool supportExtra) {
    std::map<int, VARP> varMap;
    std::vector<int> inputIndexes;
    std::vector<OpT*> extraOps;
    std::vector<OpT*> allOps;
    for (int index = 0; index < net->oplists.size(); ++index) {
        auto op = net->oplists[index].get();
        if (_isControlOp(op)) {
            extraOps.emplace_back(op);
            continue;
        }
        if (op->type == OpType_Extra && !supportExtra) {
            extraOps.emplace_back(op);
            continue;
        }
        allOps.emplace_back(op);
    }
    for (int index=0; index < allOps.size(); ++index) {
        std::set<OpT*> invalidSet;
        _create(varMap, inputIndexes, allOps, index, net, invalidSet);
    }
    std::set<VARP> outputs;
    for (auto extra : extraOps) {
        for (auto index : extra->inputIndexes) {
            if (varMap.find(index) != varMap.end()) {
                outputs.insert(varMap[index]);
            }
        }
    }
    for (auto& iter : varMap) {
        if (iter.second->linkNumber() == 0) {
            outputs.insert(iter.second);
        }
    }
    std::shared_ptr<Program> newProgram(new Program);
    Program& program = *newProgram;
    program.mVars    = varMap;
    for (auto output : outputs) {
        program.mOutputs.emplace_back(output);
    }
    if (extraOps.empty()) {
        return newProgram;
    }
    std::shared_ptr<Frame> currentFrameShared(new Frame);
    program.mFrames.emplace_back(currentFrameShared);
    auto currentFrame = currentFrameShared.get();
    for (int i = 0; i < extraOps.size(); ++i) {
        auto op = extraOps[i];
        if ((!currentFrame->whileName.empty()) && op->name.find(currentFrame->whileName) == std::string::npos) {
            currentFrame = currentFrame->parent;
        }
        if (op->type == OpType_Extra && op->main.AsExtra()->type == "Enter") {
            std::string frameName;
            for (auto& attr : op->main.AsExtra()->attr) {
                if (attr->key == "frame_name") {
                    frameName = attr->s;
                    break;
                }
            }
            if (frameName != currentFrame->name) {
                std::shared_ptr<Frame> newFrame(new Frame);
                newFrame->name   = frameName;
                int pos = frameName.size()-1;
                for (; pos > 0 ; pos--) {
                    if (frameName[pos] == '/') {
                        break;
                    }
                }
                newFrame->whileName = frameName.substr(0, pos);
                //MNN_PRINT("%s\n", newFrame->whileName.c_str());

                newFrame->parent = currentFrame;
                currentFrame->children.push_back(newFrame);
                currentFrame->body.emplace_back(nullptr);
                currentFrame = newFrame.get();
            }
        }
        currentFrame->body.emplace_back(op);
    }
    return newProgram;
}
} // namespace Express
} // namespace MNN
#ifdef BUILD_EXE
int main(int argc, const char* argv[]) {
    auto program = _splitProgram(argv[1]);
    {
        std::ofstream output("model.cpp");
        std::ofstream outputUtils("Utils.hpp");
        output << "#include <MNN/expr/Expr.hpp>\n";
        output << "#include <MNN/expr/ExprCreator.hpp>\n";
        output << "using namespace MNN::Express;\n";
        output << "int main() {\n";
        output << "auto varMap = Variable::loadMap(\"support.mnn\");\n";
        program.second->emit(program.first, output);
        program.second->emitUtils(outputUtils);
        output << "}\n";
    }
    std::vector<VARP> saves;
    for (auto iter : program.first) {
        saves.emplace_back(iter.second);
    }
    Variable::save(saves, "support.mnn");

    // program.print();
    // program.analysis();
    //_testSplit(argv[1]);
    return 0;
}
#endif
