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
    bool Leaf = false;
    float Value = 0.0;

public:
    bool IsLeaf() const;
    size_t GetChild(const TFeatureRow& data) const;
    static TDecisionTreeNode Terminate(const TPool& pool, TMask& mask);
};

using TNodes = std::vector<TDecisionTreeNode>;

class TDecisionTree {
public:
    static TDecisionTree Fit(const TPool& pool, size_t maxDepth, size_t minCount, bool verbose);
    float Predict(const TFeatureRow& data) const;

private:
    static size_t FitImpl(TDecisionTree& tree,
                          size_t depth,
                          const TPool& pool,
                          TMask& mask,
                          size_t maxDepth,
                          size_t minCount,
                          bool verbose);

public:
    TNodes Nodes;
};

void TestDecisionTree();
