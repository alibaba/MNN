#include <MNN_generated.h>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <MNN/MNNDefine.h>
using namespace MNN;
static bool reIndexTensor(std::unique_ptr<MNN::NetT>& net) {
    auto& mNet = net;
    std::map<int, int> usefulTensorIndexMap;
    std::vector<std::string> usefulTensorName;

    std::vector<bool> tensorValid(mNet->tensorName.size(), false);
    for (auto& op : mNet->oplists) {
        for (auto index : op->inputIndexes) {
            if (index < 0) {
                continue; // optional input, ignore it
            }
            tensorValid[index] = true;
        }
        for (auto index : op->outputIndexes) {
            tensorValid[index] = true;
        }
    }

    for (int i = 0; i < tensorValid.size(); ++i) {
        if (tensorValid[i]) {
            usefulTensorIndexMap.insert(std::make_pair(i, usefulTensorName.size()));
            usefulTensorName.push_back(mNet->tensorName[i]);
        }
    }

    // Re index
    for (auto& op : mNet->oplists) {
        for (int i = 0; i < op->inputIndexes.size(); ++i) {
            if (op->inputIndexes[i] < 0) {
                continue;
            }
            auto iter = usefulTensorIndexMap.find(op->inputIndexes[i]);
            op->inputIndexes[i] = iter->second;
        }
        for (int i = 0; i < op->outputIndexes.size(); ++i) {
            auto iter = usefulTensorIndexMap.find(op->outputIndexes[i]);
            op->outputIndexes[i] = iter->second;
        }
    }

    mNet->tensorName = usefulTensorName;
    for (auto iter = mNet->extraTensorDescribe.begin(); iter != mNet->extraTensorDescribe.end();) {
        auto index = (*iter)->index;
        if (usefulTensorIndexMap.find(index) == usefulTensorIndexMap.end()) {
            iter = mNet->extraTensorDescribe.erase(iter);
            continue;
        }
        (*iter)->index = usefulTensorIndexMap.find(index)->second;
        iter++;
    }
    // Check dup name and modify
    std::set<std::string> names;
    std::set<std::string> tensorNames;
    for (int i = 0; i < mNet->oplists.size(); ++i) {
        auto& op    = mNet->oplists[i];
        auto opName = op->name;
        if (opName.empty() || names.find(opName) != names.end()) {
            std::ostringstream defaultName;
            defaultName << EnumNameOpType(op->type);
            defaultName << i;
            op->name = defaultName.str();
            MNN_PRINT("%d op name is empty or dup, set to %s\n", i, op->name.c_str());
            opName = op->name;
        }
        names.insert(opName);
        for (auto output : op->outputIndexes) {
            auto origin = net->tensorName[output];
            if (origin.empty() || tensorNames.find(origin) != tensorNames.end()) {
                std::ostringstream defaultName;
                defaultName << output;
                origin                  = defaultName.str();
                net->tensorName[output] = origin;
            }
            tensorNames.insert(origin);
        }
    }
    return true;
}
static void mergeInplaceForCPU(MNN::NetT* net) {
    std::set<MNN::OpType> inplaceOps = {
        OpType_UnaryOp,
        OpType_ReLU,
        OpType_ReLU6,
        OpType_PReLU,
        OpType_Scale,
    };
    std::vector<int> useCount(net->tensorName.size(), 0);
    for (auto& op : net->oplists) {
        for (auto index : op->inputIndexes) {
            useCount[index]++;
        }
    }
    std::map<int, int> replaceIndex;
    for (int i=0; i<net->oplists.size(); ++i) {
        auto op = net->oplists[i].get();
        for (int j=0; j<op->inputIndexes.size(); ++j) {
            if (replaceIndex.find(op->inputIndexes[j]) != replaceIndex.end()) {
                op->inputIndexes[j] = replaceIndex[op->inputIndexes[j]];
            }
        }
        if (inplaceOps.find(op->type) == inplaceOps.end()) {
            continue;
        }
        if (useCount[op->inputIndexes[0]] > 1) {
            continue;
        }
        replaceIndex.insert(std::make_pair(op->outputIndexes[0], op->inputIndexes[0]));
        op->outputIndexes[0] = op->inputIndexes[0];
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_ERROR("Usage: ./mergeInplaceForCPU SRC.mnn DST.mnn\n");
        return 0;
    }
    std::unique_ptr<MNN::NetT> net;
    {
        std::ifstream inputIs(argv[1]);
        std::ostringstream inputOs;
        inputOs << inputIs.rdbuf();
        auto content = inputOs.str();
        net.reset(flatbuffers::GetRoot<MNN::Net>(content.c_str())->UnPack());
    }
    mergeInplaceForCPU(net.get());
    reIndexTensor(net);
    flatbuffers::FlatBufferBuilder builderOutput(1024);
    builderOutput.ForceDefaults(true);
    auto len = MNN::Net::Pack(builderOutput, net.get());
    builderOutput.Finish(len);
    auto sizeOutput    = builderOutput.GetSize();
    auto bufferOutput = builderOutput.GetBufferPointer();
    std::ofstream outputOs(argv[2]);
    outputOs.write((const char*)bufferOutput, sizeOutput);

    return 0;
}
