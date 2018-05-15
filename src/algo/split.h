#pragma once

#include "defines.h"

#include <vector>

class TSplit {
public:
    size_t FeatureId = 0;
    size_t bin_id = 0;
    float variance = 0.0;
};

float Mean(const TTarget& data, const std::vector<char>& mask);

float Variance(const TTarget& data, const std::vector<char>& mask);

std::pair<float, float> GetRange(const TFeature& data, const TMask& mask);

std::vector<float> GetPartition(const TFeature& data, const TMask& mask, size_t parts);

size_t GetOptimalSplit(const TFeatures& features, const TTarget& target, const TMask& mask);

