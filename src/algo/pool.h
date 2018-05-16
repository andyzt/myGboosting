#pragma once

#include "defines.h"

#include <unordered_map>

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
    //TFeatureRows Rows;
    TTarget Target;
    TNames Names;
    size_t RawFeatureCount = 0;
    size_t BinarizedFeatureCount = 0;
    size_t Size = 0;
    std::vector<std::unordered_map<std::string, size_t>> Hashes;
};

TRawPool LoadTrainingPool(const std::string& path);
TRawPool LoadTestingPool(const std::string& path, std::vector<std::unordered_map<std::string, size_t>>& hashes);
TPool ConvertPoolToBinNumbers(const TRawPool& raw, std::vector<std::vector<float>> bounds);

