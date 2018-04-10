#include "train.h"

#include "algo/defines.h"
#include "algo/pool.h"
#include "algo/split.h"
#include "algo/tree.h"

#include <iostream>
#include <numeric>
#include <sstream>

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
    node.FeatureId = GetOptimalSplit(pool.Features, pool.Target, mask);
    std::cout << "Split by feature " << node.FeatureId << std::endl;

    std::vector<size_t> maskIds;
    for (size_t id = 0; id < pool.Size; ++id) {
        if (mask[id] != 0) {
            mask[id] = pool.Features[node.FeatureId][id] >= 0.5;
            maskIds.push_back(id);
        }
    }

    tree.Nodes.push_back(node);
    auto nodeId = tree.Nodes.size() - 1;

    tree.Nodes[nodeId].Right = Train(pool, tree, mask, depth + 1, maxDepth, minCount);

    for (size_t id : maskIds) {
        mask[id] = pool.Features[node.FeatureId][id] < 0.5;
    }
    tree.Nodes[nodeId].Left = Train(pool, tree, mask, depth + 1, maxDepth, minCount);

    return nodeId;
}

TFeatureVector ReadLine(const std::string& line, const TBinarizer& binarizer) {
    TFeatureVector vector;

    std::string str;
    std::stringstream stream(line);

    size_t featureId = 0;
    while (std::getline(stream, str, ',')) {
        try {
            float value = std::stof(str);
            for (auto x : binarizer.Binarize(featureId, value)) {
                vector.push_back(x);
            }
        } catch (...) {
            for (auto x : binarizer.Binarize(featureId, str)) {
                vector.push_back(x);
            }
        }
        featureId++;
    }

    return vector;
}

void TrainMode::Run(const std::string& path) {
    std::cout << "Train" << std::endl;

    std::cout << "Loading " << path << std::endl;

    TPool pool;
    TBinarizer binarizer;

    pool = binarizer.Binarize(LoadTrainingPool(path));

    std::cout << "Done" << std::endl;
    std::cout << "Raw features: " << pool.RawFeatureCount << std::endl;
    std::cout << "Binarized features: " << pool.BinarizedFeatureCount << std::endl;
    std::cout << "Size: " << pool.Size << std::endl;

    TDecisionTree tree;
    TMask mask(pool.Size, 1);
    Train(pool, tree, mask, 0, 6, 10);

    {
        // Should be 1
        std::cout << tree.Predict(ReadLine("6.4,2.9,4.3,1.3", binarizer)) << std::endl;
    }

    {
        // Should be 0
        std::cout << tree.Predict(ReadLine("4.8,3.4,1.6,0.2", binarizer)) << std::endl;
    }
}
