#pragma once

#include "defines.h"
#include "split.h"
#include <vector>
#include <string>

int FindBin(const THistogram& histogram, float value);

std::vector<float> BuildBinBounds(const TRawFeature& data, size_t num_bins);

THistogram BuildHistogram(const TFeature& data, const TTarget& target, const TMask& mask,
                          const std::vector<float>& bounds);

std::vector<THistogram> CalcHistDifference(std::vector<THistogram> &parent_hists, std::vector<THistogram> &other);

