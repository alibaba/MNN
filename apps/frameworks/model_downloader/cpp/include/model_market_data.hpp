#pragma once

#include <string>

namespace mnn::downloader {

// Get embedded model market JSON data
const unsigned char* GetModelMarketJsonData();
unsigned int GetModelMarketJsonDataLen();

} // namespace mnn::downloader