#include "tree.h"

#include <numeric>
#include <random>
#include <set>
#include "histogram.h"

void TDecisionTreeNode::BuildHistogram(const size_t feature_id, const TFeature& data,
                                       const TTarget& target, size_t bins_size) {

    hists[feature_id].resize(bins_size);

    for (size_t idx :row_indices) {
        ++hists[feature_id][data[idx]].cumulative_cnt;
        hists[feature_id][data[idx]].cumulative_sum += target[idx];
    }

    //Building cumulative sums
    for (int i = 0; i + 1 < bins_size; ++i) {
        hists[feature_id][i+1].cumulative_cnt += hists[feature_id][i].cumulative_cnt;
        hists[feature_id][i+1].cumulative_sum += hists[feature_id][i].cumulative_sum;
    }
}

void TDecisionTreeNode::CalcHistDifference(const size_t feature_id,
                                           const THistogram& parent_hists,
                                           const THistogram& other) {

    hists[feature_id] = parent_hists;
    for (size_t bin_id = 0; bin_id < parent_hists.size();++bin_id) {
        hists[feature_id][bin_id].cumulative_cnt -= other[bin_id].cumulative_cnt;
        hists[feature_id][bin_id].cumulative_sum -= other[bin_id].cumulative_sum;
    }
}

TSplit CalcSplitHistogram(const TFeature& feature_vector,
                          const TTarget& target,
                          TNodes &Nodes,
                          const std::vector<size_t>& cur_level_nodes,
                          size_t feature_id,
                          size_t binCount) {

    TSplit split;
    double max_gain = std::numeric_limits<float>::min();

    if (cur_level_nodes.size() == 1) {
        Nodes[cur_level_nodes[0]].BuildHistogram(feature_id,
                                                 feature_vector,
                                                 target,
                                                 binCount);
    } else {
        //Let's build new histograms
        for (int left_node_idx = 0; left_node_idx < cur_level_nodes.size(); left_node_idx += 2) {
            auto &left_node = Nodes[cur_level_nodes[left_node_idx]];
            auto &right_node = Nodes[cur_level_nodes[left_node_idx + 1]];
            if (!left_node.is_empty && !right_node.is_empty) {
                if (left_node.row_indices.size() > right_node.row_indices.size()) {
                    right_node.BuildHistogram(feature_id,
                                              feature_vector,
                                              target,
                                              binCount);
                    left_node.CalcHistDifference(feature_id,
                                                 Nodes[left_node.Parent].hists[feature_id],
                                                 right_node.hists[feature_id]);
                } else {
                    left_node.BuildHistogram(feature_id,
                                             feature_vector,
                                             target,
                                             binCount);
                    right_node.CalcHistDifference(feature_id,
                                                  Nodes[left_node.Parent].hists[feature_id],
                                                  left_node.hists[feature_id]);
                }
            } else {
                //copy histograms from parent
                if (!left_node.is_empty) {
                    left_node.hists[feature_id] = Nodes[left_node.Parent].hists[feature_id];
                }
                if (!right_node.is_empty) {
                    right_node.hists[feature_id] = Nodes[left_node.Parent].hists[feature_id];
                }
            }
        }
    }

    for (size_t cur_bin =0; cur_bin <binCount; ++cur_bin) {
        double gain = 0;
        for (auto cur_node : cur_level_nodes) {
            if (Nodes[cur_node].is_empty)
                continue;
            const auto& hists = Nodes[cur_node].hists;

            double left_gain = 0.0;
            if (hists[feature_id][cur_bin].cumulative_cnt != 0) {
                double val = hists[feature_id][cur_bin].cumulative_sum;
                left_gain = val * val / hists[feature_id][cur_bin].cumulative_cnt;
            }

            double right_sum = hists[feature_id].back().cumulative_sum - hists[feature_id][cur_bin].cumulative_sum;
            float right_cnt = hists[feature_id].back().cumulative_cnt - hists[feature_id][cur_bin].cumulative_cnt;

            double right_gain = 0.0;
            if (right_cnt != 0) {
                right_gain = right_sum * right_sum / right_cnt;
            }
            gain += left_gain  + right_gain;
        }

        if (gain > max_gain) {
            split.bin_id = cur_bin;
            max_gain = gain;
        }

    }
    split.gain = max_gain;

    return split;
}

TDecisionTree TDecisionTree::FitHist(TPool& pool, size_t maxDepth, size_t minCount, float sample_rate, float lrate,
                                 std::vector<std::vector<float>>& all_bounds, bool verbose) {
    TDecisionTree tree;
    size_t pool_size = pool.Size;
    std::set<int> chosen_features;

    TDecisionTreeNode node;
    node.row_indices.reserve(static_cast<int>(pool_size * sample_rate));
    node.hists.resize(pool.Features.size());

    std::random_device rd{}; // use to seed the rng
    std::mt19937 rng{rd()}; // rng
    std::bernoulli_distribution d(sample_rate);
    for (uint32_t i = 0; i < pool_size; ++i) {
        if (d(rng))
            node.row_indices.emplace_back(i);
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
            TSplit cur_split = CalcSplitHistogram(pool.Features[feature_id],
                                                  pool.Target,
                                                  tree.Nodes,
                                                  cur_level_nodes,
                                                  feature_id,
                                                  all_bounds[feature_id].size());
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
            left_node.Parent = nodeId;
            TDecisionTreeNode right_node;
            right_node.Parent = nodeId;

            //for terminal node we copy its Value from parent
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

            //we are splitting by upper bound of a bin,
            // so for split bin_id =0 we should put 0 bin to the left
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

                double cumulative_sum = 0.0;
                for (const auto idx : tree.Nodes[nodeId].row_indices)
                    cumulative_sum += pool.Target[idx];
                tree.Nodes[nodeId].Value = cumulative_sum
                                           / tree.Nodes[nodeId].row_indices.size();
                left_node.Value = tree.Nodes[nodeId].Value;
                right_node.Value = tree.Nodes[nodeId].Value;
            }

            if (!left_node.is_empty)
                left_node.hists.resize(pool.Features.size());
            if (!right_node.is_empty)
                right_node.hists.resize(pool.Features.size());

            tree.Nodes.emplace_back(left_node);
            tree.Nodes[nodeId].Left = tree.Nodes.size() - 1;
            next_level_nodes.push_back(tree.Nodes[nodeId].Left);

            tree.Nodes.emplace_back(right_node);
            tree.Nodes[nodeId].Right = tree.Nodes.size() - 1;
            next_level_nodes.push_back(tree.Nodes[nodeId].Right);
        }

        std::swap(cur_level_nodes, next_level_nodes);
    }

    tree.values.reserve(cur_level_nodes.size());
    for (const auto& nodeId : cur_level_nodes) {
        if (tree.Nodes[nodeId].is_empty) {
            tree.values.push_back(tree.Nodes[nodeId].Value);
        } else {
            double cumulative_sum = 0.0;
            for (const auto idx : tree.Nodes[nodeId].row_indices)
                cumulative_sum += pool.Target[idx];
            tree.values.push_back(cumulative_sum / tree.Nodes[nodeId].row_indices.size());
        }
        //replacing our target by gradient of current step
        for (const auto idx : tree.Nodes[nodeId].row_indices)
            pool.Target[idx] -= lrate * tree.values.back();
    }

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
        //indexing starts from 0
        --predictions_idx[i];
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