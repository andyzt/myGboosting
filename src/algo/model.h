#pragma once

#include "binarization.h"
#include "tree.h"

class TModel {
public:
    TModel();
    explicit TModel(TBinarizer&& binarizer);

    void Fit(TPool&& pool, float rate, float iterations);
    TTarget Predict(const TPool& pool) const;
//    TTarget Predict(const TRawPool& raw) const;
    void Serialize(const std::string& filename, const TPool& pool);
    void DeSerialize(const std::string& filename,
                             std::vector<std::unordered_map<std::string, size_t>>& hashes,
                             std::vector<std::vector<float>>& splits);

private:
    float LearningRate;
    TBinarizer Binarizer;
    std::vector<TDecisionTree> Trees;
};

