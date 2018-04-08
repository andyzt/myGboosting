#pragma once

#include<string>
#include <vector>

// represents a single data column
using TFeature = std::vector<float>;

// several feature columns in a vector
using TFeatures = std::vector<TFeature>;

// target column
using TTarget = std::vector<float>;

// used for tree fitting
using TMask = std::vector<char>;

// feature names
using TNames = std::vector<std::string>;

// a single case to calculate a prediction for
using TFeatureVector = std::vector<float>;
