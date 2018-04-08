#include "train.h"

#include "algo/defines.h"
#include "algo/split.h"
#include "algo/tree.h"

#include <iostream>
#include <numeric>

TDecisionTreeNode Terminate(const TPool& pool, TMask& mask) {
    TDecisionTreeNode node;
    node.Leaf = true;
    node.Value = Mean(pool.Target, mask);
    return node;
}

size_t Train(const TPool& pool, TDecisionTree& tree, TMask& mask, size_t depth, size_t maxDepth, size_t minCount) {
    if (depth == maxDepth) {
        tree.Nodes.push_back(Terminate(pool, mask));
        std::cout << "Depth termination" << std::endl;
        return tree.Nodes.size() - 1;
    }

    auto count = size_t(std::accumulate(mask.begin(), mask.end(), 0));
    std::cout << "Count = " << count << std::endl;

    if (count < minCount) {
        tree.Nodes.push_back(Terminate(pool, mask));
        std::cout << "Count termination" << std::endl;
        return tree.Nodes.size() - 1;
    }

    TDecisionTreeNode node;
    node.Split = GetOptimalSplit(pool.Features, pool.Target, mask);
    std::cout << "Split by feature " << node.Split.FeatureId << " at " << node.Split.Value << std::endl;

    std::vector<size_t> maskIds;
    for (size_t id = 0; id < pool.Size; ++id) {
        if (mask[id] != 0) {
            mask[id] = pool.Features[node.Split.FeatureId][id] >= node.Split.Value;
            maskIds.push_back(id);
        }
    }

    tree.Nodes.push_back(node);
    auto nodeId = tree.Nodes.size() - 1;

    tree.Nodes[nodeId].Right = Train(pool, tree, mask, depth + 1, maxDepth, minCount);

    for (size_t id : maskIds) {
        mask[id] = pool.Features[node.Split.FeatureId][id] < node.Split.Value;
    }
    tree.Nodes[nodeId].Left = Train(pool, tree, mask, depth + 1, maxDepth, minCount);

    return nodeId;
}

void TrainMode::Run(TPool&& pool) {
    TDecisionTree tree;
    TMask mask(pool.Size, 1);
    Train(pool, tree, mask, 0, 3, 5);

    {
        TFeatureVector data = { 6.4, 2.9, 4.3, 1.3 };
        // Should be 1
        std::cout << tree.Predict(data) << std::endl;
    }

    {
        TFeatureVector data = { 4.8, 3.4, 1.6, 0.2 };
        // Should be 0
        std::cout << tree.Predict(data) << std::endl;
    }

}
