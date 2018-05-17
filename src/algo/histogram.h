#pragma once

#include "defines.h"
#include "split.h"
#include <vector>

std::vector<float> BuildBinBounds(const TRawFeature& data, size_t max_bins);
