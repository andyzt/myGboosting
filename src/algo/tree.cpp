#include "tree.h"

#include <numeric>
#include <random>
#include <set>



float CalcSplitValue(const TRawFeature& feature, const TTarget& target, const TMask& mask, float split, int depth) {
    auto N = target.size();

    int depth_2 = (1 << (depth+1)) + 1;

    std::vector<float> sums(depth_2, 0.0);
    std::vector<int> counters(depth_2, 0);

    for (size_t i = 0; i < N; ++i) {
        if (!mask[i]) {
            continue;
        }

        if (feature[i] >= split) {
            ++counters[mask[i] << 1];
            sums[mask[i] << 1] += target[i];
        } else {
            ++counters[(mask[i] << 1) - 1];
            sums[(mask[i] << 1) - 1] += target[i];
        }

    }

    std::vector<float> means(depth_2, 0.0);

    for (int j = 1; j < depth_2; ++j) {
        if (counters[j])
            means[j] = sums[j] / counters[j];
    }

    std::vector<float> variances(depth_2, 0.0);
    for (size_t i = 0; i < N; ++i) {
        if (!mask[i]) {
            continue;
        }
        if (feature[i] >= split) {
            float val = target[i] - means[mask[i] << 1];
            variances[mask[i] << 1] += val * val;
        } else {
            float val = target[i] - means[(mask[i] << 1) - 1];
            variances[(mask[i] << 1) - 1] += val * val;
        }

    }

    float total_variance = 0.0;

    for (int k = 1; k < depth_2; ++k) {
        total_variance += variances[k]*counters[k];
    }

    return total_variance;
}

TDecisionTree TDecisionTree::Fit(const TRawPool& pool, size_t maxDepth, size_t minCount, float sample_rate, int max_bins,
                                 std::vector<std::vector<float>>& bounds, bool verbose) {
    TDecisionTree tree;
    size_t pool_size = pool.RawFeatures[0].size();
    TMask mask(pool_size, 1);

    std::random_device rd{}; // use to seed the rng
    std::mt19937 rng{rd()}; // rng
    std::bernoulli_distribution d(sample_rate);
    for (int i = 0; i < pool_size; ++i) {
        mask[i] = d(rng);
    }

    std::set<int> chosen_features;


    for (int depth = 0; depth < maxDepth; ++depth) {
        float minVariance = std::numeric_limits<float>::max();
        int minFeatureId;
        float minSplit;
        for (int featureId = 0; featureId < pool.RawFeatures.size(); ++featureId) {
            if (chosen_features.find(featureId) != chosen_features.end())
                continue;
            for (int i = 0; i + 1 < bounds[featureId].size(); ++i) {
                float curVariance = CalcSplitValue(pool.RawFeatures[featureId], pool.Target, mask,
                                                   bounds[featureId][i], depth);

                if (curVariance < minVariance) {
                    minVariance = curVariance;
                    minFeatureId = featureId;
                    minSplit = bounds[featureId][i];
                }
            }
        }

        chosen_features.insert(minFeatureId);
        tree.splits.emplace_back(minFeatureId, minSplit);

        for (size_t i = 0; i < pool_size; ++i) {
            if (!mask[i]) {
                continue;
            }

            if (pool.RawFeatures[minFeatureId][i] >= minSplit) {
                mask[i] <<= 1;
            } else {
                mask[i] = (mask[i] << 1) - 1;
            }

        }
    }


    int depth_2 = (1 << maxDepth)+1;
    std::vector<float> sums(depth_2, 0.0);
    std::vector<int> counters(depth_2, 0);

    for (size_t i = 0; i < pool_size; ++i) {
        if (!mask[i]) {
            continue;
        }
            ++counters[mask[i]];
            sums[mask[i]] += pool.Target[i];

    }

    tree.values.resize(depth_2, 0.0);
    for (int j = 0; j < depth_2; ++j) {
        if (counters[j])
            tree.values[j] = sums[j] / counters[j];
    }

    return tree;
}

/*
float TDecisionTree::Predict(const TFeatureRow& data) const {
    int idx = 1;
    for (const auto& it : splits) {
        if (data[it.first] >= it.second) {
            idx <<= 1;
        } else {
            idx = (idx << 1) - 1;
        }
    }
    return values[idx];
}
*/

std::vector<int> TDecisionTree::GetPredictionIndices(TRawPool& pool) const {
    std::vector<int> predictions_idx(pool.Target.size(), 1);
    for (int i = 0; i < pool.Target.size(); ++i) {
        for (const auto& it : splits) {
            if (pool.RawFeatures[it.first][i] >= it.second) {
                predictions_idx[i] <<= 1;
            } else {
                predictions_idx[i] = (predictions_idx[i] << 1) - 1;
            }
        }
    }

    return predictions_idx;
}

void TDecisionTree::AddPredict(TRawPool& pool, float lrate, TTarget& predictions) const {
    std::vector<int> predictions_idx = GetPredictionIndices(pool);

    for (int j = 0; j < pool.Target.size(); ++j) {
        predictions[j] += lrate * values[predictions_idx[j]];
    }
}

void TDecisionTree::ModifyTargetByPredict(TRawPool&& pool, float lrate) const {
    std::vector<int> predictions_idx = GetPredictionIndices(pool);

    for (int j = 0; j < pool.Target.size(); ++j) {
        pool.Target[j] -= lrate * values[predictions_idx[j]];
    }
}

/*
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
*/