#pragma once

#include<string>
#include <vector>

// represents a single data column
using TRawFeature = std::vector<float>;

// raw features
using TRawFeatures = std::vector<TRawFeature>;

// binarized form of a feature vector
using TFeature = std::vector<char>;

// several feature columns in a vector
using TFeatures = std::vector<TFeature>;

// target column
using TTarget = std::vector<float>;

// used for tree fitting
using TMask = std::vector<char>;

// feature names
using TNames = std::vector<std::string>;

// a single case to calculate a prediction for
using TRawFeatureVector = std::vector<float>;

// binarized form
using TFeatureVector = std::vector<char>;

struct HistogramBin {
    size_t cnt = 0;
    float target_sum = 0;
    float upper_bound;
};

// histogram
using THistogram = std::vector<HistogramBin>;
