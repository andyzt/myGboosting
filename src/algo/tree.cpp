#include "tree.h"

#include <numeric>
#include <random>
#include <set>
#include "histogram.h"

TSplit CalcSplitHistogram(TNodes &Nodes, const std::vector<size_t>& cur_level_nodes,
                          size_t featureId, size_t binCount) {

    TSplit split;
    float minVariance = std::numeric_limits<float>::max();

    for (size_t bin =0; bin <binCount; ++bin) {
        float variance = 0;
        for (auto cur_node : cur_level_nodes) {
            const auto& hists = Nodes[cur_node].hists;

            int sum_count = 0;
            float target_sum = 0.0;

            for (auto bin : hists[featureId]) {
                sum_count += bin.cnt;
                target_sum += bin.target_sum;
            }

            int cur_count = 0;
            float cur_target_sum = 0;


            if (hists[featureId][bin].cnt == 0)
                continue;

            cur_count += hists[featureId][bin].cnt;
            cur_target_sum += hists[featureId][bin].target_sum;

            float mean = cur_target_sum / float(cur_count);
            float left_variance = 0.0;
            for (size_t i = 0; i <= bin; ++i) {
                if (hists[featureId][i].cnt == 0)
                    continue;
                float val = (hists[featureId][i].target_sum / float(hists[featureId][i].cnt) - mean);
                left_variance += val*val;
            }

            if (sum_count == cur_count)
                break;

            mean = (target_sum - cur_target_sum) / float(sum_count - cur_count);
            float right_variance = 0.0;
            for (size_t i = bin + 1; i < hists[featureId].size(); ++i) {
                if (hists[featureId][i].cnt == 0)
                    continue;
                float val = (hists[featureId][i].target_sum / float(hists[featureId][i].cnt) - mean);
                right_variance += val*val;
            }

            variance += left_variance*float(cur_count) +
                        right_variance*float(sum_count - cur_count);

            Nodes[cur_node].l_count[featureId] = cur_count;
            Nodes[cur_node].r_count[featureId] = sum_count - cur_count;

        }

        if (variance < minVariance) {
            split.bin_id = bin;
            minVariance = variance;
        }

    }

    split.variance = minVariance;
    return split;
}

TDecisionTree TDecisionTree::FitHist(const TRawPool& pool, size_t maxDepth, size_t minCount, float sample_rate,
                                 std::vector<std::vector<float>>& bounds, bool verbose) {
    TDecisionTree tree;
    size_t pool_size = pool.RawFeatures[0].size();


    std::set<int> chosen_features;

    TDecisionTreeNode node;
    node.l_count.resize(pool.RawFeatures.size());
    node.r_count.resize(pool.RawFeatures.size());

    std::random_device rd{}; // use to seed the rng
    std::mt19937 rng{rd()}; // rng
    std::bernoulli_distribution d(sample_rate);
    for (uint32_t i = 0; i < pool_size; ++i) {
        if (d(rng))
            node.row_indices.push_back(i);
    }

    for (size_t feature_id =0; feature_id <pool.RawFeatures.size(); ++feature_id) {
        node.hists.push_back(BuildHistogram(pool.RawFeatures[feature_id], pool.Target, node.row_indices,
                                            bounds[feature_id]));
    }

    tree.Nodes.emplace_back(node);
    std::vector<size_t> cur_level_nodes({0});

    for (size_t depth = 0; depth < maxDepth; ++depth) {
        float minVariance = std::numeric_limits<float>::max();
        int minFeatureId;
        int bin_id;

        for (size_t feature_id = 0; feature_id < pool.RawFeatures.size(); ++feature_id) {
            if (chosen_features.find(feature_id) != chosen_features.end())
                continue;
            TSplit cur_split = CalcSplitHistogram(tree.Nodes, cur_level_nodes, feature_id, bounds[feature_id].size() - 1);

            if (cur_split.variance < minVariance) {
                minVariance = cur_split.variance;
                minFeatureId = feature_id;
                bin_id = cur_split.bin_id;
            }
        }

        chosen_features.insert(minFeatureId);

        //TODO rebuild to bin indexes instead of Value
        float minSplit = tree.Nodes[cur_level_nodes[0]].Value;
        tree.splits.emplace_back(minFeatureId, minSplit);
        std::cout << minFeatureId << " : " << bin_id << std::endl;


        std::vector<size_t> next_level_nodes;
        for (const auto& nodeId : cur_level_nodes) {

            //Let's build left child node with new histograms
            TDecisionTreeNode left_node;
            left_node.l_count.resize(pool.RawFeatures.size());
            left_node.r_count.resize(pool.RawFeatures.size());
            left_node.Parent = nodeId;

            //Let's build left child node with new histograms
            TDecisionTreeNode right_node;
            right_node.l_count.resize(pool.RawFeatures.size());
            right_node.r_count.resize(pool.RawFeatures.size());
            right_node.Parent = nodeId;

            for(const auto& cur_idx : tree.Nodes[nodeId].row_indices) {
                if (pool.RawFeatures[minFeatureId][cur_idx] >= minSplit) {
                    right_node.row_indices.push_back(cur_idx);
                } else {
                    left_node.row_indices.push_back(cur_idx);
                }
            }

            //Let's build new histograms
            if (tree.Nodes[nodeId].l_count[minFeatureId] > tree.Nodes[nodeId].r_count[minFeatureId]) {
                for (size_t feature_id =0; feature_id <pool.RawFeatures.size(); ++feature_id) {
                    right_node.hists.push_back(BuildHistogram(pool.RawFeatures[feature_id], pool.Target,
                                                              right_node.row_indices, bounds[feature_id]));
                }

                left_node.hists = CalcHistDifference(tree.Nodes[nodeId].hists, right_node.hists);

            } else {
                for (size_t feature_id =0; feature_id <pool.RawFeatures.size(); ++feature_id) {
                    left_node.hists.push_back(BuildHistogram(pool.RawFeatures[feature_id], pool.Target,
                                                             left_node.row_indices, bounds[feature_id]));
                }

                right_node.hists = CalcHistDifference(tree.Nodes[nodeId].hists, left_node.hists);
            }

            tree.Nodes.emplace_back(left_node);
            tree.Nodes[nodeId].Left = tree.Nodes.size() - 1;
            next_level_nodes.push_back(tree.Nodes[nodeId].Left);

            tree.Nodes.emplace_back(right_node);
            tree.Nodes[nodeId].Right = tree.Nodes.size() - 1;
            next_level_nodes.push_back(tree.Nodes[nodeId].Right);
        }

        cur_level_nodes = std::move(next_level_nodes);
    }

    //first value is dummy
    tree.values.push_back(0);
    for (const auto& nodeId : cur_level_nodes) {
        //TODO: get Target from parent
        if (tree.Nodes[nodeId].row_indices.size() == 0) {
            tree.values.push_back(0);
            continue;
        }
        float sum_target = 0;
        for (const auto idx : tree.Nodes[nodeId].row_indices)
            sum_target += pool.Target[idx];

        tree.values.push_back(sum_target / tree.Nodes[nodeId].row_indices.size());
    }

    return tree;
}

std::vector<int> TDecisionTree::GetPredictionIndices(TRawPool& pool) const {
    std::vector<int> predictions_idx(pool.Target.size(), 1);
    for (size_t i = 0; i < pool.Target.size(); ++i) {
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

    for (size_t j = 0; j < pool.Target.size(); ++j) {
        predictions[j] += lrate * values[predictions_idx[j]];
    }
}

void TDecisionTree::ModifyTargetByPredict(TRawPool&& pool, float lrate) const {
    std::vector<int> predictions_idx = GetPredictionIndices(pool);

    for (size_t j = 0; j < pool.Target.size(); ++j) {
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