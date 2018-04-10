#include "histogram.h"

#include <iostream>

int FindBin(const THistogram& histogram, float value) {

    int low = 0;                    // нижняя граница
    int up = histogram.size() - 1;    // верхняя граница

    while (low <= up) {
        int m = (low + up) / 2;
        if (histogram[m].upper_bound == value) {
            low = m + 1;
            break;
        }
        if (histogram[m].upper_bound < value)
            low = m + 1;
        if (histogram[m].upper_bound > value)
            up = m - 1;
    }

    return low;

}

std::vector<float> BuildBinBounds(const TFeature& data, size_t num_bins) {
    TFeature sorted_data(data);
    std::sort(sorted_data.begin(), sorted_data.end());

    float avg_bin_size = data.size() / num_bins;
    std::vector<float> bounds;
    size_t cur_bin_size = 0;
    float cur_bound_value = sorted_data[0];
    std::cout << "Low: " << cur_bound_value;
    bounds.push_back(cur_bound_value);

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
    bounds.push_back(std::numeric_limits<float>::max());

    return bounds;
}

THistogram BuildHistogram(const TFeature& data, const TTarget& target, const TMask& mask,
                          const std::vector<float>& bounds) {

    THistogram histogram;

    for (size_t i = 1; i < bounds.size(); ++i) {
        HistogramBin new_bin;
        new_bin.upper_bound = bounds[i];

        histogram.push_back(new_bin);
    }

    for (size_t i = 0; i < data.size(); ++i) {
        if (!mask[i]) {
            continue;
        }
        int bin = FindBin(histogram, data[i]);
        ++histogram[bin].cnt;
        histogram[bin].target_sum += target[i];
    }
    return histogram;
}

std::vector<THistogram> CalcHistDifference(std::vector<THistogram> &parent_hists, std::vector<THistogram> &other) {
    std::vector<THistogram> new_hists(parent_hists);

    for (size_t i = 0; i < parent_hists.size();++i)
        for (size_t j =0; j < parent_hists[i].size(); ++j) {
            new_hists[i][j].cnt -= other[i][j].cnt;
            new_hists[i][j].target_sum -= other[i][j].target_sum;
        }
    return new_hists;
}

