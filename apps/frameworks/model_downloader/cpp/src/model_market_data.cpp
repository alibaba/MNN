#include "model_market_data.hpp"

namespace mnn::downloader {

// Embedded model market JSON data
#include "model_market_json_data.inc"
const unsigned int model_market_json_data_len = sizeof(model_market_json_data);

const unsigned char* GetModelMarketJsonData() {
    return model_market_json_data;
}

unsigned int GetModelMarketJsonDataLen() {
    return model_market_json_data_len;
}

} // namespace mnn::downloader