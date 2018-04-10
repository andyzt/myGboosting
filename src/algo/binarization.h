#pragma once

#include "defines.h"
#include "pool.h"

#include <vector>

class TBinarizer {
public:
    TPool Binarize(TRawPool&& raw);
    TFeatureVector Binarize(size_t featureId, const std::string& value) const;
    TFeatureVector Binarize(size_t featureId, float value) const;

private:
    TFeatures BinarizeFloatFeature(const TRawFeature& data, std::vector<float> splits);
    TFeatures BinarizeCatFeature(const TRawFeature& data, size_t cats);

private:
    std::vector<size_t> BinarizedToRaw;
    std::vector<std::vector<float>> Splits;
    std::vector<std::unordered_map<std::string, size_t>> Hashes;
};