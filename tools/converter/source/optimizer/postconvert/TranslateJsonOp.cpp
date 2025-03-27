//
//  TranslateJsonOp.cpp
//  MNNConverter
//
//  Created by MNN on 2025/02/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/MNNDefine.h>
#include "../PostTreatUtils.hpp"
#include "../../common/Json2Flatbuffer.hpp"
#include <map>
#include <set>
using namespace MNN;

class TranslateJsonOp : public PostConverter {
public:
    static bool isJsonOp(const MNN::OpT* op) {
        if (op->type != OpType_Extra || op->main.type != OpParameter_Extra) {
            return false;
        }
        auto extra = op->main.AsExtra();
        if (extra->engine != "MNN" || extra->type != "JSON") {
            return false;
        }
        return true;
    }
    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        for (int i=0; i<net->oplists.size(); ++i) {
            auto op = net->oplists[i].get();
            if (!isJsonOp(op)) {
                continue;
            }
            auto extra = op->main.AsExtra();
            std::string content;
            for (auto& attr : extra->attr) {
                if (attr->key == "main") {
                    content = attr->s;
                    continue;
                }
            }
            rapidjson::Document document;
            document.Parse(content.c_str());
            if (document.HasParseError()) {
                MNN_ERROR("Invalid json Op in TranslateJsonOp\n");
                continue;
            }
            auto object = document.GetObject();
            flatbuffers::FlatBufferBuilder builder;
            auto offset = Json2Flatbuffer::writeJsonToFlatbuffer(  MNN::OpTypeTable(), builder, object);
            builder.Finish(offset);
            // Cache input and output
            auto inputIndex = std::move(op->inputIndexes);
            auto outputIndex = std::move(op->outputIndexes);
            auto name = std::move(op->name);
            net->oplists[i].reset(flatbuffers::GetRoot<Op>(builder.GetBufferPointer())->UnPack());
            net->oplists[i]->inputIndexes = std::move(inputIndex);
            net->oplists[i]->outputIndexes = std::move(outputIndex);
            net->oplists[i]->name = std::move(name);
        }
        return true;
    }
};
static PostConverterRegister<TranslateJsonOp> __l("TranslateJsonOp");
