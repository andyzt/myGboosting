#pragma once

#include "binarization.h"
#include "tree.h"

class TModel {
public:
    TModel();

    void Fit(TRawPool& raw_pool, const Config& config);
    TTarget Predict(TPool& pool) const;
    TTarget PredictOnTestData(const TRawPool& raw) const;
    void Serialize(const std::string& filename);
    void DeSerialize(const std::string& filename);

private:
    float LearningRate;
    std::vector<TDecisionTree> Trees;
    std::vector<std::vector<float>> upper_bounds;
};

