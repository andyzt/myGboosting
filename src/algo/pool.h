#pragma once

#include "defines.h"

#include <unordered_map>

class TRawPool {
public:
    TRawFeatures RawFeatures;
    TTarget Target;
};

class TPool {
public:
    TFeatures Features;
    TTarget Target;
    size_t Size = 0;
   };

TRawPool LoadTrainingPool(const std::string& path);
TRawPool LoadTestingPool(const std::string& path);
TPool ConvertPoolToBinNumbers(const TRawPool& raw, std::vector<std::vector<float>>& bounds);

