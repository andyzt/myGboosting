#include "histogram.h"

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

uint8_t FindBin(const std::vector<float>& bounds, float value) {

    int low = 0;                    // нижняя граница
    int up = bounds.size() - 1;    // верхняя граница


    while (low <= up) {
        int m = (low + up) / 2;
        if (bounds[m] == value) {
            low = m + 1;
            break;
        }
        if (bounds[m] < value)
            low = m + 1;
        if (bounds[m] > value)
            up = m - 1;
    }

    return low;

}

std::vector<float> BuildBinBounds(const TRawFeature& data, size_t num_bins) {
    TRawFeature sorted_data(data);
    std::sort(sorted_data.begin(), sorted_data.end());

    float avg_bin_size = data.size() / num_bins;
    std::vector<float> bounds;
    size_t cur_bin_size = 0;
    float cur_bound_value = sorted_data[0];
    std::cout << "Low: " << cur_bound_value;
    //we build an array of upper bounds so we don't need minimum there
    //bounds.push_back(cur_bound_value);

    for (const auto value : sorted_data) {
        if ((cur_bound_value - value) < 1e-6 && (value - cur_bound_value) < 1e-6) {
            ++cur_bin_size;
            continue;
        }
        cur_bound_value = value;
        if (cur_bin_size >= avg_bin_size) {
            bounds.push_back(cur_bound_value);
            std::cout << "  " << cur_bound_value;
            cur_bin_size = 0;
        }
        ++cur_bin_size;
    }
    std::cout << " High: " << cur_bound_value << std::endl;
    //to include maximum in the last bin we increase its upper bound to +inf
    bounds.push_back(std::numeric_limits<float>::max());

    return bounds;
}

THistogram BuildHistogram(const TFeature& data, const TTarget& target, const std::vector<uint32_t> row_indices,
                          size_t bins_size) {


    THistogram histogram(bins_size);

    for (size_t idx :row_indices) {
        ++histogram[data[idx]].cnt;
        histogram[data[idx]].target_sum += target[idx];
    }

    //Building cumulative sums
    histogram[0].cumulative_cnt = histogram[0].cnt;
    histogram[0].cumulative_sum = histogram[0].target_sum;
    for (int i = 0; i + 1 < bins_size; ++i) {
        histogram[i+1].cumulative_cnt = histogram[i].cumulative_cnt + histogram[i+1].cnt;
        histogram[i+1].cumulative_sum = histogram[i].cumulative_sum + histogram[i+1].target_sum;
    }
    return histogram;
}

std::vector<THistogram> CalcHistDifference(std::vector<THistogram> &parent_hists, std::vector<THistogram> &other) {
    std::vector<THistogram> new_hists(parent_hists);

    for (size_t i = 0; i < parent_hists.size();++i)
        for (size_t j =0; j < parent_hists[i].size(); ++j) {
            new_hists[i][j].cnt -= other[i][j].cnt;
            new_hists[i][j].target_sum -= other[i][j].target_sum;
            new_hists[i][j].cumulative_cnt -= other[i][j].cumulative_cnt;
            new_hists[i][j].cumulative_sum -= other[i][j].cumulative_sum;
        }
    return new_hists;
}

