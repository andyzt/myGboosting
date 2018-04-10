#include <vector>
#include <iostream>
#include "split.h"
#include <numeric>

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

TSplit GetOptimalSplitHistogram(std::vector<THistogram> &hists, const TFeatures& features, const TTarget& target) {

    TSplit split;
    float minVariance = std::numeric_limits<float>::max();

    for (size_t featureId = 0; featureId < features.size(); ++featureId) {

        int sum_count = 0;
        float target_sum = 0.0;

        for (auto bin : hists[featureId]) {
            sum_count += bin.cnt;
            target_sum += bin.target_sum;
        }

        int cur_count = 0;
        float cur_target_sum = 0;

        for (size_t bin =0; bin <hists[featureId].size(); ++bin) {
            if (hists[featureId][bin].cnt == 0)
                continue;

            cur_count += hists[featureId][bin].cnt;
            cur_target_sum += hists[featureId][bin].target_sum;

            float mean = cur_target_sum / float(cur_count);
            float left_variance = 0.0;
            for (size_t i = 0; i <= bin; ++i) {
                if (hists[featureId][i].cnt == 0)
                    continue;
                float val = (hists[featureId][i].target_sum / float(hists[featureId][i].cnt) - mean);
                left_variance += val*val;
            }

            if (sum_count == cur_count)
                break;

            mean = (target_sum - cur_target_sum) / float(sum_count - cur_count);
            float right_variance = 0.0;
            for (size_t i = bin + 1; i < hists[featureId].size(); ++i) {
                if (hists[featureId][i].cnt == 0)
                    continue;
                float val = (hists[featureId][i].target_sum / float(hists[featureId][i].cnt) - mean);
                right_variance += val*val;
            }

            auto variance = left_variance*float(cur_count) +
                    right_variance*float(sum_count - cur_count);


            if (variance < minVariance) {
                split.FeatureId = featureId;
                split.Value = hists[featureId][bin].upper_bound;
                split.l_count = cur_count;
                split.r_count = sum_count - cur_count;

                minVariance = variance;
            }
        }
    }

    return split;
}

