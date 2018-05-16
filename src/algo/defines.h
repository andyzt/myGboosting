#pragma once

#include<string>
#include <vector>

// represents a single data column
using TRawFeature = std::vector<float>;

// raw features
using TRawFeatures = std::vector<TRawFeature>;

// binarized form of a feature
using TFeature = std::vector<u_int8_t>;

// several feature columns in a vector
using TFeatures = std::vector<TFeature>;

// target column
using TTarget = std::vector<float>;

// used for tree fitting
using TMask = std::vector<int16_t>;

// feature names
using TNames = std::vector<std::string>;

// a single case to calculate a prediction for
using TRawFeatureRow = std::vector<float>;

// binarized form
using TFeatureRow = std::vector<char>;

// several rows
using TFeatureRows = std::vector<TFeatureRow>;

struct HistogramBin {
    size_t cnt = 0;
    size_t cumulative_cnt = 0;
    float target_sum = 0;
    float cumulative_sum = 0;

    float upper_bound;
};

// histogram
using THistogram = std::vector<HistogramBin>;
