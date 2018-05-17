#include "tree.h"

#include <numeric>
#include <random>
#include <set>
#include "histogram.h"

TSplit CalcSplitHistogram(TNodes &Nodes, const std::vector<size_t>& cur_level_nodes,
                          size_t featureId, size_t binCount) {

    TSplit split;
    double max_gain = std::numeric_limits<float>::min();


    for (size_t cur_bin =0; cur_bin <binCount; ++cur_bin) {
        float gain = 0;
        for (auto cur_node : cur_level_nodes) {
            if (Nodes[cur_node].is_empty)
                continue;

            const auto& hists = Nodes[cur_node].hists;

            double left_gain = 0.0;
            //std::cout << hists[featureId][cur_bin].cumulative_cnt << ' ';
            if (hists[featureId][cur_bin].cumulative_cnt != 0) {
                float val = hists[featureId][cur_bin].cumulative_sum;
                //std::cout << hists[featureId][cur_bin].cumulative_cnt << ' ';
                left_gain = val * val / hists[featureId][cur_bin].cumulative_cnt;
            }

            float right_sum = hists[featureId].back().cumulative_sum - hists[featureId][cur_bin].cumulative_sum;
            float right_cnt = hists[featureId].back().cumulative_cnt - hists[featureId][cur_bin].cumulative_cnt;

            double right_gain = 0.0;
            if (right_cnt != 0) {
                right_gain = right_sum * right_sum / right_cnt;
            }

            gain += (left_gain  + right_gain); // / (hists[featureId][cur_bin].cumulative_cnt + right_cnt);

            Nodes[cur_node].l_count[featureId] = hists[featureId][cur_bin].cumulative_cnt;
            Nodes[cur_node].r_count[featureId] = right_cnt;

        }

        if (gain > max_gain) {
            split.bin_id = cur_bin;
            max_gain = gain;
        }

    }
    split.gain = max_gain;


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
        node.hists.push_back(BuildHistogram(pool.Features[feature_id], pool.Target, node.row_indices,
                                            all_bounds[feature_id].size()));
    }

    tree.Nodes.emplace_back(node);
    std::vector<size_t> cur_level_nodes({0});
    std::vector<size_t> next_level_nodes;

    for (size_t depth = 0; depth < maxDepth; ++depth) {
        double max_gain = std::numeric_limits<float>::min();
        int best_feature;
        int bin_id;

        for (size_t feature_id = 0; feature_id < pool.Features.size(); ++feature_id) {
            if (chosen_features.find(feature_id) != chosen_features.end())
                continue;
            TSplit cur_split = CalcSplitHistogram(tree.Nodes, cur_level_nodes, feature_id, all_bounds[feature_id].size());
            //std::cout << std::endl;
            //std::cout << feature_id << " :: " << cur_split.gain << std::endl;
            if (cur_split.gain > max_gain) {
                max_gain = cur_split.gain;
                best_feature = feature_id;
                bin_id = cur_split.bin_id;
            }
        }

        chosen_features.insert(best_feature);

        tree.splits.emplace_back(best_feature, bin_id);
        //std::cout << best_feature << " : " << bin_id << std::endl;


        next_level_nodes.clear();
        next_level_nodes.reserve(2*cur_level_nodes.size());
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

            if (tree.Nodes[nodeId].is_empty) {
                left_node.Value = tree.Nodes[nodeId].Value;
                left_node.is_empty = true;
                right_node.Value = tree.Nodes[nodeId].Value;
                right_node.is_empty = true;

                tree.Nodes.emplace_back(left_node);
                tree.Nodes[nodeId].Left = tree.Nodes.size() - 1;
                next_level_nodes.push_back(tree.Nodes[nodeId].Left);

                tree.Nodes.emplace_back(right_node);
                tree.Nodes[nodeId].Right = tree.Nodes.size() - 1;
                next_level_nodes.push_back(tree.Nodes[nodeId].Right);

                continue;
            }

            //we are splitting by upper bound of a bin, so for split bin_id =0 we should put 0 bin to the left
            //that's why we have > here (not >=)
            for(const auto& cur_idx : tree.Nodes[nodeId].row_indices) {
                if (pool.Features[best_feature][cur_idx] > bin_id) {
                    right_node.row_indices.push_back(cur_idx);
                } else {
                    left_node.row_indices.push_back(cur_idx);
                }
            }

            left_node.is_empty = left_node.row_indices.size() < 1;
            right_node.is_empty = right_node.row_indices.size() < 1;

            if (left_node.is_empty || right_node.is_empty) {

                tree.Nodes[nodeId].Value = tree.Nodes[nodeId].hists[0].back().cumulative_sum
                                           / tree.Nodes[nodeId].row_indices.size();
                left_node.Value = tree.Nodes[nodeId].Value;
                right_node.Value = tree.Nodes[nodeId].Value;
            }

            //Let's build new histograms
            if (!left_node.is_empty && !right_node.is_empty) {
                if (tree.Nodes[nodeId].l_count[best_feature] > tree.Nodes[nodeId].r_count[best_feature]) {
                    for (size_t feature_id = 0; feature_id < pool.Features.size(); ++feature_id) {
                        right_node.hists.push_back(BuildHistogram(pool.Features[feature_id], pool.Target,
                                                                  right_node.row_indices,
                                                                  all_bounds[feature_id].size()));
                    }

                    left_node.hists = CalcHistDifference(tree.Nodes[nodeId].hists, right_node.hists);

                } else {

                    for (size_t feature_id = 0; feature_id < pool.Features.size(); ++feature_id) {
                        left_node.hists.push_back(BuildHistogram(pool.Features[feature_id], pool.Target,
                                                                 left_node.row_indices,
                                                                 all_bounds[feature_id].size()));
                    }


                    right_node.hists = CalcHistDifference(tree.Nodes[nodeId].hists, left_node.hists);
                }
            } else {
                //copy histograms from parent
                if (!left_node.is_empty) {
                        left_node.hists = tree.Nodes[nodeId].hists;
                }

                if (!right_node.is_empty) {
                    right_node.hists = tree.Nodes[nodeId].hists;
                }
            }

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
    tree.values.reserve(cur_level_nodes.size()+1);
    tree.values.push_back(0);
    for (const auto& nodeId : cur_level_nodes) {

        if (tree.Nodes[nodeId].is_empty) {
            tree.values.push_back(tree.Nodes[nodeId].Value);
    //        std::cout << tree.values.back() << ' ';
            continue;
        }

        tree.values.push_back(tree.Nodes[nodeId].hists[0].back().cumulative_sum / tree.Nodes[nodeId].row_indices.size());
    //    std::cout << tree.values.back() << ' ';
    }
    //std::cout << std::endl;

    return tree;
}

std::vector<int> TDecisionTree::GetPredictionIndices(TPool& pool) const {
    std::vector<int> predictions_idx(pool.Target.size(), 1);
    for (size_t i = 0; i < pool.Target.size(); ++i) {
        for (const auto& it : splits) {
            if (pool.Features[it.first][i] > it.second) {
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