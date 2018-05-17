#include "histogram.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

//bin bounds search is based on function GreedyFindBin from
//https://github.com/Microsoft/LightGBM/blob/master/src/io/bin.cpp
std::vector<float> BuildBinBounds(const TRawFeature& data, size_t max_bins) {

    TRawFeature feature_copy(data);
    std::sort(feature_copy.begin(), feature_copy.end());

    std::vector<float> distinct_values = {feature_copy[0]};
    std::vector<uint32_t> counts = {1};
    for (uint32_t i = 1; i < feature_copy.size(); ++i) {
        if (feature_copy[i] == distinct_values.back()) {
            ++counts.back();
        } else {
            distinct_values.push_back(feature_copy[i]);
            counts.push_back(1);
        }
    }
    size_t row_count = feature_copy.size();
    size_t num_distinct_values = distinct_values.size();

    std::vector<float> bin_upper_bound;
    if (num_distinct_values <= max_bins) {
        for (int i = 0; i < num_distinct_values - 1; ++i) {
            bin_upper_bound.push_back((distinct_values[i] + distinct_values[i + 1]) / 2.0);
        }
        bin_upper_bound.push_back(std::numeric_limits<float>::max());
    } else {
        double mean_bin_size = static_cast<double>(row_count) / max_bins;

        // mean size for one bin
        int rest_bin_cnt = max_bins;
        int rest_sample_cnt = static_cast<int>(row_count);
        std::vector<bool> is_big_count_value(num_distinct_values, false);
        for (int i = 0; i < num_distinct_values; ++i) {
            if (counts[i] >= mean_bin_size) {
                is_big_count_value[i] = true;
                --rest_bin_cnt;
                rest_sample_cnt -= counts[i];
            }
        }
        mean_bin_size = static_cast<double>(rest_sample_cnt) / rest_bin_cnt;
        std::vector<float> upper_bounds(max_bins, std::numeric_limits<float>::infinity());
        std::vector<float> lower_bounds(max_bins, std::numeric_limits<float>::infinity());

        int bin_cnt = 0;
        lower_bounds[bin_cnt] = distinct_values[0];
        int cur_cnt_inbin = 0;
        for (int i = 0; i < num_distinct_values - 1; ++i) {
            if (!is_big_count_value[i]) {
                rest_sample_cnt -= counts[i];
            }
            cur_cnt_inbin += counts[i];
            // need a new bin
            if (is_big_count_value[i] || cur_cnt_inbin >= mean_bin_size ||
                (is_big_count_value[i + 1] && cur_cnt_inbin >= std::max(1.0, mean_bin_size * 0.5f))) {
                upper_bounds[bin_cnt] = distinct_values[i];
                ++bin_cnt;
                lower_bounds[bin_cnt] = distinct_values[i + 1];
                if (bin_cnt >= max_bins - 1) { break; }
                cur_cnt_inbin = 0;
                if (!is_big_count_value[i]) {
                    --rest_bin_cnt;
                    mean_bin_size = rest_sample_cnt / static_cast<double>(rest_bin_cnt);
                }
            }
        }
        ++bin_cnt;

        // update bin upper bound
        bin_upper_bound.clear();
        for (int i = 0; i < bin_cnt - 1; ++i) {
            auto val = (upper_bounds[i] + lower_bounds[i + 1]) / 2.0;
            if (bin_upper_bound.empty() || val > bin_upper_bound.back()) {
                bin_upper_bound.push_back(val);
            }
        }
        // last bin upper bound
        bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
    }

    return bin_upper_bound;
}



