#pragma once

#include "binarization.h"
#include "tree.h"

class TModel {
public:
    TModel();

    void Fit(TRawPool& raw_pool, float rate, float iterations, float sample_rate,
             size_t depth, size_t min_leaf_count, size_t max_bins);
    TTarget Predict(TPool& pool) const;
//    TTarget Predict(const TRawPool& raw) const;
    void Serialize(const std::string& filename, const TPool& pool);
    void DeSerialize(const std::string& filename,
                             std::vector<std::unordered_map<std::string, size_t>>& hashes,
                             std::vector<std::vector<float>>& splits);

private:
    float LearningRate;
    std::vector<TDecisionTree> Trees;
};

