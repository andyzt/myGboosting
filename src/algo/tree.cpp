#include "tree.h"

bool TDecisionTreeNode::IsLeaf() const {
    return Leaf;
}

size_t TDecisionTreeNode::GetChild(const TFeatureVector& data) const {
    if (data[FeatureId] >= 0.5) {
        return Right;
    } else {
        return Left;
    }
}

float TDecisionTree::Predict(const TFeatureVector& data) {
    size_t nodeId = 0;

    while (!Nodes[nodeId].IsLeaf()) {
        nodeId = Nodes[nodeId].GetChild(data);
    }

    return Nodes[nodeId].Value;
}

size_t TDecisionTreeNode::GetChildPool(const TFeatures& data, const size_t row_num) const {
    if (data[FeatureId][row_num] >= 0.5) {
        return Right;
    } else {
        return Left;
    }
}

float TDecisionTree::PredictPool(const TFeatures& data, const size_t row_num) const {
    size_t nodeId = 0;

    while (!Nodes[nodeId].IsLeaf()) {
        nodeId = Nodes[nodeId].GetChildPool(data, row_num);
    }

    return Nodes[nodeId].Value;
}

void TestDecisionTree() {
    TDecisionTree tree;

    {
        TDecisionTreeNode node;
        node.FeatureId = 1;
        node.Left = 1;
        node.Right = 2;
        tree.Nodes.push_back(node);
    }

    {
        TDecisionTreeNode node;
        node.FeatureId = 0;
        node.Left = 3;
        node.Right = 4;
        tree.Nodes.push_back(node);
    }

    {
        TDecisionTreeNode node;
        node.Leaf = true;
        node.Value = 10;
        tree.Nodes.push_back(node);
    }

    {
        TDecisionTreeNode node;
        node.Leaf = true;
        node.Value = 0;
        tree.Nodes.push_back(node);
    }

    {
        TDecisionTreeNode node;
        node.Leaf = true;
        node.Value = -10;
        tree.Nodes.push_back(node);
    }

    TFeatureVector data = {0, 1};

    std::cout << tree.Predict(data) << std::endl;
}
