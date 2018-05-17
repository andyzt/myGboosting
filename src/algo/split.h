#pragma once

#include "defines.h"

#include <vector>

class TSplit {
public:
    size_t FeatureId = 0;
    size_t bin_id = 0;
    double gain = 0.0;
};

float Mean(const TTarget& data, const std::vector<char>& mask);

float Variance(const TTarget& data, const std::vector<char>& mask);

