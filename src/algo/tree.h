#pragma once

#include "defines.h"
#include "split.h"
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
    TSplit split;
    std::vector<THistogram> hists;
    std::vector<uint32_t> row_indices;
    std::vector<size_t> l_count;
    std::vector<size_t> r_count;

public:
    bool IsLeaf() const;
    size_t GetChild(const TFeatureRow& data) const;
    static TDecisionTreeNode Terminate(const TPool& pool, TMask& mask);
};

using TNodes = std::vector<TDecisionTreeNode>;

class TDecisionTree {
public:

    static TDecisionTree FitHist(const TPool& pool, size_t maxDepth, size_t minCount, float sample_rate,
                             std::vector<std::vector<float>>& all_bounds, bool verbose);
    //float Predict(const TFeatureRow& data) const;
    void AddPredict(TPool& pool, float lrate, TTarget& predictions) const;
    void ModifyTargetByPredict(TPool&& pool, float lrate) const;

private:
    std::vector<int> GetPredictionIndices(TPool& pool) const;
    static size_t FitImpl(TDecisionTree& tree,
                          size_t depth,
                          const TPool& pool,
                          TMask& mask,
                          size_t maxDepth,
                          size_t minCount,
                          bool verbose);

public:
    TNodes Nodes;
    std::vector<std::pair<int, u_int8_t>> splits;
    std::vector<float> values;
};

void TestDecisionTree();

TSplit CalcSplitHistogram(TNodes &Nodes, const std::vector<size_t>& cur_level_nodes,
                          size_t featureId, size_t binCount);