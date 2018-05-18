#pragma once

#include "defines.h"
#include "pool.h"

#include <iostream>

class TDecisionTreeNode {
public:
    size_t FeatureId = 0;
    size_t Left = 0;
    size_t Right = 0;
    size_t Parent = 0;


    bool is_empty = false;
    float Value = 0.0;

    std::vector<THistogram> hists;
    std::vector<uint32_t> row_indices;

public:
    void BuildHistogram(const size_t feature_id, const TFeature& data,
                        const TTarget& target, size_t bins_size);
    void CalcHistDifference(const size_t feature_id,
                            const THistogram& parent_hists,
                            const THistogram& other);

};

using TNodes = std::vector<TDecisionTreeNode>;

class TDecisionTree {
public:

    static TDecisionTree FitHist(TPool& pool,
                                 size_t maxDepth,
                                 size_t minCount,
                                 float sample_rate,
                                 std::vector<std::vector<float>>& all_bounds, bool verbose);

    //float Predict(const TFeatureRow& data) const;
    void AddPredict(TPool& pool, float lrate, TTarget& predictions) const;
    void ModifyTargetByPredict(TPool& pool, float lrate) const;


public:

    std::vector<std::pair<int, u_int8_t>> splits;
    std::vector<float> values;
};

TSplit CalcSplitHistogram(const TFeature& feature_vector,
                          const TTarget& target,
                          TNodes &Nodes,
                          const std::vector<size_t>& cur_level_nodes,
                          size_t feature_id,
                          size_t binCount);