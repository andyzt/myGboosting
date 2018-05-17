#pragma once

#include <algo/pool.h>
#include "algo/defines.h"
#include "algo/tree.h"

class TrainMode {
public:
    static void Run(const std::string& path, const int iterations, const float lrate, const int depth,
                    const float sample_rate, const int max_bins, const int min_leaf_count,
                    const std::string& output_file, const bool verbose);
};


