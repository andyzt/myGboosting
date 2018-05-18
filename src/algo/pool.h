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

TRawPool LoadPool(const Config& config);
TPool ConvertPoolToBinNumbers(const TRawPool& raw, const std::vector<std::vector<float>>& bounds);

