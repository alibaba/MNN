#pragma once

#include <string>

namespace mnncli {

// Get embedded model market JSON data
const unsigned char* GetModelMarketJsonData();
unsigned int GetModelMarketJsonDataLen();

} // namespace mnncli