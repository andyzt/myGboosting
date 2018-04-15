#pragma once

#include <algo/pool.h>
#include "algo/defines.h"
#include "algo/tree.h"

class TrainMode {
public:
    static void Run(const std::string &path);

    static std::vector<float> MakePredictions(const TPool &pool, const std::vector<TDecisionTree>& tree_vector);
    static std::vector<float> SingleTreePredictions(const TPool &pool, const TDecisionTree& tree);
};


