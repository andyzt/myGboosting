#include <vector>
#include <iostream>
#include "split.h"
#include <numeric>
#include <limits>

float Mean(const TTarget& data, const TMask& mask) {
    float mean = 0.0;
    float count = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        mean += mask[i] * data[i];
        count += mask[i];
    }
    mean /= count;
    return mean;
}

float Variance(const TTarget& data, const TMask& mask) {
    auto mean = Mean(data, mask);

    float variance = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        float val = mask[i]*(data[i] - mean);
        variance += val*val;
    }

    return variance;
}

size_t GetOptimalSplit(const TFeatures& features, const TTarget& target, const TMask& mask) {
    auto N = target.size();

    size_t split = 0;
    float minVariance = std::numeric_limits<float>::max();

    for (size_t featureId = 0; featureId < features.size(); ++featureId) {
        const auto& data = features[featureId];
        TMask mask1 = mask;
        TMask mask2 = mask;

        size_t count1 = 0;
        size_t count2 = 0;
        size_t total = 0;
        for (size_t i = 0; i < N; ++i) {
            if (!mask[i]) {
                continue;
            }

            mask1[i] = (data[i] >= 0.5);
            mask2[i] = !mask1[i];

            total++;
            count1 += mask1[i];
            count2 += mask2[i];
        }

        auto variance = Variance(target, mask1)*float(count1) +
                        Variance(target, mask2)*float(count2);

        if (variance < minVariance) {
            split = featureId;
            minVariance = variance;
        }
    }

    return split;
}



