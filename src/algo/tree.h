#pragma once

#include "defines.h"
#include "split.h"

#include <iostream>

class TDecisionTreeNode {
public:
    size_t FeatureId;
    size_t Left = 0;
    size_t Right = 0;
    bool Leaf = false;
    float Value = 0.0;

public:
    size_t GetChild(const TFeatureVector& data) const;
    bool IsLeaf() const;
};

using TNodes = std::vector<TDecisionTreeNode>;

class TDecisionTree {
public:
    float Predict(const TFeatureVector& data);

public:
    TNodes Nodes;
};


void TestDecisionTree();
