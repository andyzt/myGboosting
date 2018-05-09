#pragma once

#include <algo/pool.h>
#include "algo/defines.h"
#include "algo/tree.h"

class TrainMode {
public:
    static void Run(const std::string& path, const int iterations, const float rate, const int depth,
                    const float sample_rate, const std::string& output_file);
};


