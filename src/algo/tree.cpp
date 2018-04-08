#include "tree.h"

bool TDecisionTreeNode::IsLeaf() const {
    return Leaf;
}

size_t TDecisionTreeNode::GetChild(const TFeatureVector& data) const {
    if (data[Split.FeatureId] >= Split.Value) {
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

void TestDecisionTree() {
    TDecisionTree tree;

    {
        TDecisionTreeNode node;
        node.Split.FeatureId = 1;
        node.Split.Value = 5.0;
        node.Left = 1;
        node.Right = 2;
        tree.Nodes.push_back(node);
    }

    {
        TDecisionTreeNode node;
        node.Split.FeatureId = 0;
        node.Split.Value = 0.0;
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
        node.Value = -10;
        tree.Nodes.push_back(node);
    }

    {
        TDecisionTreeNode node;
        node.Leaf = true;
        node.Value = 0;
        tree.Nodes.push_back(node);
    }

    TFeatureVector data = {4, 3};

    std::cout << tree.Predict(data) << std::endl;
}
