#pragma once

#include "defines.h"
#include "split.h"
#include <vector>

uint8_t FindBin(const std::vector<float>& bounds, float value);

std::vector<float> BuildBinBounds(const TRawFeature& data, size_t num_bins);

THistogram BuildHistogram(const TFeature& data, const TTarget& target, const std::vector<uint32_t> row_indices,
                          size_t bins_size);

std::vector<THistogram> CalcHistDifference(std::vector<THistogram> &parent_hists, std::vector<THistogram> &other);

