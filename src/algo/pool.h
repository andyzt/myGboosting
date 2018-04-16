#pragma once

#include <unordered_map>
#include "defines.h"

class TRawPool {
public:
    TNames Names;
    TRawFeatures RawFeatures;
    TTarget Target;
    std::vector<std::pair<float, float>> Ranges;
    std::vector<std::unordered_map<std::string, size_t>> Hashes;
};

class TPool {
public:
    TFeatures Features;
    TTarget Target;
    TNames Names;
    float learning_rate;
    size_t RawFeatureCount = 0;
    size_t BinarizedFeatureCount = 0;
    size_t Size = 0;
};

TRawPool LoadTrainingPool(const std::string& path);
