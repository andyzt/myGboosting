#pragma once

#include "defines.h"
#include "histogram.h"

#include <vector>

class TSplit {
public:
    size_t FeatureId = 0;
    float Value = 0.0;
    size_t l_count;
    size_t r_count;
};

float Mean(const TFeature& data, const std::vector<char>& mask);

float Variance(const TFeature& data, const std::vector<char>& mask);

std::pair<float, float> GetRange(const TFeature& data, const TMask& mask);

std::vector<float> GetPartition(const TFeature& data, const TMask& mask, size_t parts);

TSplit GetOptimalSplit(const TFeatures& features, const TTarget& target, const TMask& mask);

TSplit GetOptimalSplitHistogram(std::vector<THistogram> &hists, const TFeatures& features, const TTarget& target);
