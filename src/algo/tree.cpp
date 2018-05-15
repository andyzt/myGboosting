#include "tree.h"

#include <numeric>
#include <random>
#include <set>
#include "histogram.h"

TSplit CalcSplitHistogram(TNodes &Nodes, const std::vector<size_t>& cur_level_nodes,
                          size_t featureId, size_t binCount) {

    TSplit split;
    float minVariance = std::numeric_limits<float>::max();


    for (size_t cur_bin =0; cur_bin <binCount; ++cur_bin) {
        float variance = 0;
        for (auto cur_node : cur_level_nodes) {
            if (Nodes[cur_node].is_empty)
                continue;

            const auto& hists = Nodes[cur_node].hists;


            float left_variance = 0.0;
            if (hists[featureId][cur_bin].cumulative_cnt != 0) {
                float mean = hists[featureId][cur_bin].cumulative_sum / float(hists[featureId][cur_bin].cumulative_cnt);

                for (size_t i = 0; i <= cur_bin; ++i) {
                    if (hists[featureId][i].cnt == 0)
                        continue;
                    float val = (hists[featureId][i].target_sum / float(hists[featureId][i].cnt)) - mean;
                    left_variance += val * val;
                }
            }

            float right_sum = hists[featureId].back().cumulative_sum - hists[featureId][cur_bin].cumulative_sum;
            float right_cnt = hists[featureId].back().cumulative_cnt - hists[featureId][cur_bin].cumulative_cnt;

            float right_variance = 0.0;
            if (right_cnt != 0) {
                float mean = (right_sum) / right_cnt;


                for (size_t i = cur_bin + 1; i < hists[featureId].size(); ++i) {
                    if (hists[featureId][i].cnt == 0)
                        continue;
                    float val = (hists[featureId][i].target_sum / float(hists[featureId][i].cnt)) - mean;
                    right_variance += val * val;
                }
            }

            //if (featureId == 2) {
            //    std::cout << "Left: " << left_variance << " Right: " << right_variance << std::endl;
            //}

            variance += left_variance * hists[featureId][cur_bin].cumulative_cnt +
                            right_variance * right_cnt;

            Nodes[cur_node].l_count[featureId] = hists[featureId][cur_bin].cumulative_cnt;
            Nodes[cur_node].r_count[featureId] = right_cnt;

        }

        if (variance < minVariance) {
            split.bin_id = cur_bin;
            minVariance = variance;
        }

    }
    split.variance = minVariance;


    return split;
}

TDecisionTree TDecisionTree::FitHist(const TPool& pool, size_t maxDepth, size_t minCount, float sample_rate,
                                 std::vector<std::vector<float>>& all_bounds, bool verbose) {
    TDecisionTree tree;
    size_t pool_size = pool.Size;


    std::set<int> chosen_features;

    TDecisionTreeNode node;
    node.l_count.resize(pool.Features.size());
    node.r_count.resize(pool.Features.size());

    std::random_device rd{}; // use to seed the rng
    std::mt19937 rng{rd()}; // rng
    std::bernoulli_distribution d(sample_rate);
    for (uint32_t i = 0; i < pool_size; ++i) {
        if (d(rng))
            node.row_indices.push_back(i);
    }

    for (size_t feature_id =0; feature_id <pool.Features.size(); ++feature_id) {
        //std::cout << all_bounds[feature_id].size() << std::endl;
        node.hists.push_back(BuildHistogram(pool.Features[feature_id], pool.Target, node.row_indices,
                                            all_bounds[feature_id].size()));
    }

    tree.Nodes.emplace_back(node);
    std::vector<size_t> cur_level_nodes({0});
    std::vector<size_t> next_level_nodes;

    for (size_t depth = 0; depth < maxDepth; ++depth) {
        float minVariance = std::numeric_limits<float>::max();
        int minFeatureId;
        int bin_id;

        for (size_t feature_id = 0; feature_id < pool.Features.size(); ++feature_id) {
            //if (chosen_features.find(feature_id) != chosen_features.end())
            //    continue;
            TSplit cur_split = CalcSplitHistogram(tree.Nodes, cur_level_nodes, feature_id, all_bounds[feature_id].size());
            //std::cout << feature_id << " :: " << cur_split.variance << std::endl;
            if (cur_split.variance < minVariance) {
                minVariance = cur_split.variance;
                minFeatureId = feature_id;
                bin_id = cur_split.bin_id;
            }
        }

        chosen_features.insert(minFeatureId);

        tree.splits.emplace_back(minFeatureId, bin_id);
        //std::cout << minFeatureId << " : " << bin_id << std::endl;


        next_level_nodes.clear();
        for (const auto& nodeId : cur_level_nodes) {

            //Let's build left child node with new histograms
            TDecisionTreeNode left_node;
            left_node.l_count.resize(pool.Features.size());
            left_node.r_count.resize(pool.Features.size());
            left_node.Parent = nodeId;

            //Let's build left child node with new histograms
            TDecisionTreeNode right_node;
            right_node.l_count.resize(pool.Features.size());
            right_node.r_count.resize(pool.Features.size());
            right_node.Parent = nodeId;

            //we are splitting by upper bound of a bin, so for split bin_id =0 we should put 0 bin to the left
            //that's why we have > here (not >=)
            for(const auto& cur_idx : tree.Nodes[nodeId].row_indices) {
                if (pool.Features[minFeatureId][cur_idx] > bin_id) {
                    right_node.row_indices.push_back(cur_idx);
                } else {
                    left_node.row_indices.push_back(cur_idx);
                }
            }

            if (left_node.row_indices.empty())
                left_node.is_empty = true;
            if (right_node.row_indices.empty())
                right_node.is_empty = true;

            //Let's build new histograms
            if (tree.Nodes[nodeId].l_count[minFeatureId] > tree.Nodes[nodeId].r_count[minFeatureId]) {
                for (size_t feature_id =0; feature_id <pool.Features.size(); ++feature_id) {
                    right_node.hists.push_back(BuildHistogram(pool.Features[feature_id], pool.Target,
                                                              right_node.row_indices, all_bounds[feature_id].size()));
                }

                left_node.hists = CalcHistDifference(tree.Nodes[nodeId].hists, right_node.hists);

            } else {
                for (size_t feature_id =0; feature_id <pool.Features.size(); ++feature_id) {
                    left_node.hists.push_back(BuildHistogram(pool.Features[feature_id], pool.Target,
                                                             left_node.row_indices, all_bounds[feature_id].size()));
                }

                right_node.hists = CalcHistDifference(tree.Nodes[nodeId].hists, left_node.hists);
            }

            /*
            for (int i = 0; i < pool.Features.size(); ++i) {
                for (auto hist : left_node.hists[i])
                    std::cout << hist.cnt << ' ';
                std::cout << std::endl;
            }

            std::cout << std::endl;
            for (int i = 0; i < pool.Features.size(); ++i) {
                for (auto hist : right_node.hists[i])
                    std::cout << hist.cnt << ' ';
                std::cout << std::endl;
            }
            */

            tree.Nodes.emplace_back(left_node);
            tree.Nodes[nodeId].Left = tree.Nodes.size() - 1;
            next_level_nodes.push_back(tree.Nodes[nodeId].Left);

            tree.Nodes.emplace_back(right_node);
            tree.Nodes[nodeId].Right = tree.Nodes.size() - 1;
            next_level_nodes.push_back(tree.Nodes[nodeId].Right);
        }

         std::swap(cur_level_nodes, next_level_nodes);
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

std::vector<int> TDecisionTree::GetPredictionIndices(TPool& pool) const {
    std::vector<int> predictions_idx(pool.Target.size(), 1);
    for (size_t i = 0; i < pool.Target.size(); ++i) {
        for (const auto& it : splits) {
            if (pool.Features[it.first][i] >= it.second) {
                predictions_idx[i] <<= 1;
            } else {
                predictions_idx[i] = (predictions_idx[i] << 1) - 1;
            }
        }
    }

    return predictions_idx;
}

void TDecisionTree::AddPredict(TPool& pool, float lrate, TTarget& predictions) const {
    std::vector<int> predictions_idx = GetPredictionIndices(pool);

    for (size_t j = 0; j < pool.Target.size(); ++j) {
        predictions[j] += lrate * values[predictions_idx[j]];
    }
}

void TDecisionTree::ModifyTargetByPredict(TPool&& pool, float lrate) const {
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