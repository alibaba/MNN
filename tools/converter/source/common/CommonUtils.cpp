#include <fstream>
#include <sstream>
#include <google/protobuf/util/json_util.h>
#include "CommonUtils.hpp"
#include "core/MNNFileUtils.h"
using namespace std;
void CommonKit::loadCompress(modelConfig& modelPath) {
    bool overwrite = false;
    bool readProto = false;
    modelPath.compressInfo = new PostTreatContext;
    do {
        auto compressFileName = modelPath.compressionParamsFile;
        if (compressFileName != "") {
            if (!MNNFileExist(compressFileName.c_str())) {
                MNN_PRINT("CompressFileName not exist, create it\n");
                overwrite = true;
                break;
            }
            string jsonSuffix = "json";
            string suffix = compressFileName.substr(compressFileName.find_last_of('.') + 1);
            if (jsonSuffix.compare(suffix) != 0) { // protobuf.bin file
                std::fstream input(compressFileName.c_str(), std::ios::in | std::ios::binary);
                if (!modelPath.compressInfo->proto.ParseFromIstream(&input)) {
                    MNN_ERROR("Failed to parse compression pipeline proto.\n");
                } else {
                    readProto = true;
                }
            } else {
                readProto = CommonKit::json2protobuf(compressFileName.c_str(), nullptr, &modelPath.compressInfo->proto);
            }
        }
    } while (false);
    modelPath.compressInfo->read = readProto;
    modelPath.compressInfo->write = overwrite;
}
bool CommonKit::FileIsExist(std::string path) {
    return MNNFileExist(path.c_str());
}
bool CommonKit::protobuf2json(const char* jsonFile, const MNN::Compression::Pipeline* pipeline) {
    google::protobuf::util::JsonOptions options;
    options.add_whitespace = true;
    std::string output;
    auto status = google::protobuf::util::MessageToJsonString(*pipeline, &output, options);
    if (status.code() != google::protobuf::util::status_internal::StatusCode::kOk) {
        FUNC_PRINT_ALL(status.message().ToString().c_str(), s);
        return false;
    }
    {
        std::ofstream out(jsonFile);
        if (out.fail()) {
            MNN_ERROR("Dump json error: Can't open %s\n", jsonFile);
            return false;
        }
        out << output;
    }
    return true;
}

bool CommonKit::json2protobuf(const char* jsonFile, const char* protoFile, MNN::Compression::Pipeline* pipeline) {
    {
        std::ifstream fileNames(jsonFile);
        if (fileNames.fail()) {
            return false;
        }
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        google::protobuf::util::JsonParseOptions options;
        options.ignore_unknown_fields = true;
        auto status = google::protobuf::util::JsonStringToMessage(outputStr, pipeline, options);
        if (status.code() != google::protobuf::util::status_internal::StatusCode::kOk) {
            FUNC_PRINT_ALL(status.message().ToString().c_str(), s);
            return false;
        }
    }
    return true;
}

