#include "tree.h"

#include <numeric>
#include <random>

bool TDecisionTreeNode::IsLeaf() const {
    return Leaf;
}

size_t TDecisionTreeNode::GetChild(const TFeatureRow& data) const {
    if (data[FeatureId] >= 0.5) {
        return Right;
    } else {
        return Left;
    }
}

TDecisionTreeNode TDecisionTreeNode::Terminate(const TPool& pool, TMask& mask) {
    TDecisionTreeNode node;
    node.Leaf = true;
    node.Value = Mean(pool.Target, mask);
    return node;
}

TDecisionTree TDecisionTree::Fit(const TPool& pool, size_t maxDepth, size_t minCount, float sample_rate, bool verbose) {
    TDecisionTree tree;
    TMask mask(pool.Size, 1);

    std::random_device rd{}; // use to seed the rng
    std::mt19937 rng{rd()}; // rng
    std::bernoulli_distribution d(sample_rate);
    for (int i = 0; i < pool.Size; ++i) {
        mask[i] = d(rng);
    }

    FitImpl(tree, 0, pool, mask, maxDepth, minCount, verbose);
    return tree;

}

size_t TDecisionTree::FitImpl(TDecisionTree& tree,
                              size_t depth,
                              const TPool& pool,
                              TMask& mask,
                              size_t maxDepth,
                              size_t minCount,
                              bool verbose) {
    if (depth == maxDepth) {
        tree.Nodes.push_back(TDecisionTreeNode::Terminate(pool, mask));
        if (verbose) {
            std::cout << "Depth termination" << std::endl;
        }
        return tree.Nodes.size() - 1;
    }

    auto count = size_t(std::accumulate(mask.begin(), mask.end(), 0));
    if (verbose) {
        std::cout << "Count = " << count << std::endl;
    }

    if (count < minCount) {
        tree.Nodes.push_back(TDecisionTreeNode::Terminate(pool, mask));
        if (verbose) {
            std::cout << "Count termination" << std::endl;
        }
        return tree.Nodes.size() - 1;
    }

    TDecisionTreeNode node;
    node.FeatureId = GetOptimalSplit(pool.Features, pool.Target, mask);
    if (verbose) {
        std::cout << "Split by feature " << node.FeatureId << std::endl;
    }

    std::vector<size_t> maskIds;
    for (size_t id = 0; id < pool.Size; ++id) {
        if (mask[id] != 0) {
            mask[id] = pool.Features[node.FeatureId][id] >= 0.5;
            maskIds.push_back(id);
        }
    }

    tree.Nodes.push_back(node);
    auto nodeId = tree.Nodes.size() - 1;

    tree.Nodes[nodeId].Right = FitImpl(tree, depth + 1, pool, mask, maxDepth, minCount, verbose);

    for (size_t id : maskIds) {
        mask[id] = pool.Features[node.FeatureId][id] < 0.5;
    }
    tree.Nodes[nodeId].Left = FitImpl(tree, depth + 1, pool, mask, maxDepth, minCount, verbose);

    return nodeId;
}

float TDecisionTree::Predict(const TFeatureRow& data) const {
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

    TFeatureRow data = {0, 1};

    std::cout << tree.Predict(data) << std::endl;
}
