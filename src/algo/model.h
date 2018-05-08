#pragma once

#include "binarization.h"
#include "tree.h"

class TModel {
public:
    explicit TModel(TBinarizer&& binarizer);

    void Fit(TPool&& pool, float rate, float iterations);
    TTarget Predict(const TPool& pool) const;
//    TTarget Predict(const TRawPool& raw) const;
    void Serialize(const std::string& filename);
    void DeSerialize(const std::string& filename);

private:
    float LearningRate;
    TBinarizer Binarizer;
    std::vector<TDecisionTree> Trees;
};

