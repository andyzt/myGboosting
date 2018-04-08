#include <vector>
#include <iostream>
#include <limits>
#include "split.h"

float Mean(const TFeature& data, const TMask& mask) {
    float mean = 0.0;
    float count = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        mean += mask[i] * data[i];
        count += mask[i];
    }
    mean /= count;
    return mean;
}

float Variance(const TFeature& data, const TMask& mask) {
    auto mean = Mean(data, mask);

    float variance = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        float val = mask[i]*(data[i] - mean);
        variance += val*val;
    }

    return variance;
}

std::pair<float, float> GetRange(const TFeature& data, const TMask& mask) {
    auto min = std::numeric_limits<float>::max();
    auto max = std::numeric_limits<float>::min();

    for (size_t i = 0; i < data.size(); ++i) {
        if (!mask[i]) {
            continue;
        }

        auto x = data[i];
        min = std::min(x, min);
        max = std::max(x, max);
    }

    return std::make_pair(min, max);
}

std::vector<float> GetPartition(const TFeature& data, const TMask& mask, size_t parts) {
    std::vector<float> partition;
    auto [min, max] = GetRange(data, mask);
    for (size_t i = 1; i < parts; ++i) {
        partition.push_back(min + i*(max - min)/float(parts));
    }
    return partition;
}

TSplit GetOptimalSplit(const TFeatures& features, const TTarget& target, const TMask& mask) {
    auto N = target.size();

    TSplit split;
    float minVariance = std::numeric_limits<float>::max();

    for (size_t featureId = 0; featureId < features.size(); ++featureId) {
        const auto& data = features[featureId];
        for (auto value : GetPartition(data, mask, 10)) {
            TMask mask1 = mask;
            TMask mask2 = mask;

            size_t count1 = 0;
            size_t count2 = 0;
            size_t total = 0;
            for (size_t i = 0; i < N; ++i) {
                if (!mask[i]) {
                    continue;
                }

                mask1[i] = (data[i] >= value);
                mask2[i] = !mask1[i];

                total++;
                count1 += mask1[i];
                count2 += mask2[i];
            }

            auto variance = Variance(target, mask1)*float(count1) +
                            Variance(target, mask2)*float(count2);

            if (variance < minVariance) {
                split.FeatureId = featureId;
                split.Value = value;
                minVariance = variance;
            }
        }
    }

    return split;
}

