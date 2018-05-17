#include "tree.h"

#include <numeric>
#include <set>
#include "histogram.h"
#include <random>

/*
void TDecisionTreeNode::BuildAllHistograms(const TFeatures& features,
                                       const TTarget& target) {


    for (size_t idx :row_indices) {
        float cur_target = target[idx];
        for (int feature_id = 0; feature_id < features.size(); ++feature_id) {
            uint8_t bin_num = features[feature_id][idx];
            ++hists[feature_id][bin_num].cumulative_cnt;
            hists[feature_id][bin_num].cumulative_sum += cur_target;
        }
    }

    //Building cumulative sums
    for (int feature_id = 0; feature_id < features.size(); ++feature_id) {
        for (int i = 0; i + 1 < hists[feature_id].size(); ++i) {
            hists[feature_id][i + 1].cumulative_cnt += hists[feature_id][i].cumulative_cnt;
            hists[feature_id][i + 1].cumulative_sum += hists[feature_id][i].cumulative_sum;
        }
    }

}
*/

void TDecisionTreeNode::BuildHistogram(const size_t feature_id, const TFeature& data,
                                       const TTarget& target, size_t bins_size) {

    hists[feature_id].resize(bins_size);

    for (size_t idx :row_indices) {
        ++hists[feature_id][data[idx]].cumulative_cnt;
        hists[feature_id][data[idx]].cumulative_sum += target[idx];
    }

    //Building cumulative sums
    for (size_t i = 0; i + 1 < bins_size; ++i) {
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
                          TNodes& cur_level_nodes,
                          const TNodes& parent_level_nodes,
                          size_t feature_id,
                          size_t binCount) {

    TSplit split;
    double max_gain = std::numeric_limits<float>::min();

    if (cur_level_nodes.size() == 1) {
        cur_level_nodes[0].BuildHistogram(feature_id,
                                                 feature_vector,
                                                 target,
                                                 binCount);
    } else {
        //Let's build new histograms
        for (size_t left_node_idx = 0; left_node_idx < cur_level_nodes.size(); left_node_idx += 2) {
            auto &left_node = cur_level_nodes[left_node_idx];
            auto &right_node = cur_level_nodes[left_node_idx + 1];

            if (!left_node.is_empty && !right_node.is_empty) {
                if (left_node.row_indices.size() > right_node.row_indices.size()) {
                    right_node.BuildHistogram(feature_id,
                                              feature_vector,
                                              target,
                                              binCount);
                    left_node.CalcHistDifference(feature_id,
                                                 parent_level_nodes[left_node.Parent].hists[feature_id],
                                                 right_node.hists[feature_id]);
                } else {
                    left_node.BuildHistogram(feature_id,
                                             feature_vector,
                                             target,
                                             binCount);
                    right_node.CalcHistDifference(feature_id,
                                                  parent_level_nodes[left_node.Parent].hists[feature_id],
                                                  left_node.hists[feature_id]);
                }
            } else {

                //copy histograms from parent
                if (!left_node.is_empty) {
                    left_node.hists[feature_id] = parent_level_nodes[left_node.Parent].hists[feature_id];
                }
                if (!right_node.is_empty) {
                    right_node.hists[feature_id] = parent_level_nodes[left_node.Parent].hists[feature_id];
                }
            }
        }
    }

    for (size_t cur_bin =0; cur_bin <binCount; ++cur_bin) {
        double gain = 0;
        for (auto& cur_node : cur_level_nodes) {
            if (cur_node.is_empty)
                continue;

            double left_gain = 0.0;
            if (cur_node.hists[feature_id][cur_bin].cumulative_cnt != 0) {
                double val = cur_node.hists[feature_id][cur_bin].cumulative_sum;
                left_gain = val * val / cur_node.hists[feature_id][cur_bin].cumulative_cnt;
            }

            double right_sum = cur_node.hists[feature_id].back().cumulative_sum -
                    cur_node.hists[feature_id][cur_bin].cumulative_sum;
            float right_cnt = cur_node.hists[feature_id].back().cumulative_cnt -
                    cur_node.hists[feature_id][cur_bin].cumulative_cnt;

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


TDecisionTree TDecisionTree::FitHist(TPool& pool,
                                     size_t maxDepth,
                                     size_t minCount,
                                     float sample_rate,
                                     std::vector<std::vector<float>>& all_bounds,
                                     bool verbose) {


    std::vector<uint32_t> row_indices;
    row_indices.reserve(static_cast<int>(pool.Size * sample_rate));

    if (sample_rate >= 1.0) {
        row_indices = std::vector<uint32_t>(pool.Size);
        std::iota(row_indices.begin(), row_indices.end(), 0);
    } else {
        std::random_device rd{}; // use to seed the rng
        std::mt19937 rng{rd()}; // rng
        //std::bernoulli_distribution d(sample_rate);
        std::uniform_real_distribution<double> d(0.0,1.0);

        for (size_t i = 0; i < pool.Size; ++i) {
            if (d(rng) < sample_rate)
                row_indices.emplace_back(i);
        }
    }

    TDecisionTree tree;
    std::set<int> chosen_features;

    TNodes cur_level_nodes;
    TNodes next_level_nodes;

    TDecisionTreeNode node;
    node.row_indices = row_indices;
    node.hists.resize(pool.Features.size());
    /*
    for (int feature_id = 0; feature_id < pool.Features.size(); ++feature_id) {
        node.hists[feature_id].resize(all_bounds[feature_id].size());
    }
    node.BuildAllHistograms(pool.Features, pool.Target);
*/

    cur_level_nodes.emplace_back(node);


    for (size_t depth = 0; depth < maxDepth; ++depth) {
        double max_gain = std::numeric_limits<float>::min();
        int best_feature;
        int bin_id;

        //rename to parents after std::swap with cur_level on prev iteration
        const auto& parent_level_nodes = next_level_nodes;
        for (size_t feature_id = 0; feature_id < pool.Features.size(); ++feature_id) {
            if (chosen_features.find(feature_id) != chosen_features.end())
                continue;
            TSplit cur_split = CalcSplitHistogram(pool.Features[feature_id],
                                                  pool.Target,
                                                  cur_level_nodes,
                                                  parent_level_nodes,
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
        next_level_nodes.reserve(2 * cur_level_nodes.size());
        for (size_t node_id = 0; node_id < cur_level_nodes.size(); ++ node_id) {

            //Let's build left child node with new histograms
            TDecisionTreeNode left_node;
            left_node.Parent = node_id;
            TDecisionTreeNode right_node;
            right_node.Parent = node_id;

            //for terminal node we copy its Value from parent
            if (cur_level_nodes[node_id].is_empty) {
                left_node.Value = cur_level_nodes[node_id].Value;
                left_node.is_empty = true;
                right_node.Value = cur_level_nodes[node_id].Value;
                right_node.is_empty = true;

                next_level_nodes.emplace_back(left_node);
                next_level_nodes.emplace_back(right_node);
                continue;
            }


            left_node.row_indices.reserve(cur_level_nodes[node_id].row_indices.size());
            right_node.row_indices.reserve(cur_level_nodes[node_id].row_indices.size());
            //we are splitting by upper bound of a bin,
            // so for split bin_id =0 we should put 0 bin to the left
            //that's why we have > here (not >=)
            for(const auto& cur_idx : cur_level_nodes[node_id].row_indices) {
                if (pool.Features[best_feature][cur_idx] > bin_id) {
                    right_node.row_indices.push_back(cur_idx);
                } else {
                    left_node.row_indices.push_back(cur_idx);
                }
            }

            left_node.is_empty = left_node.row_indices.size() < minCount;
            right_node.is_empty = right_node.row_indices.size() < minCount;

            if (left_node.is_empty || right_node.is_empty) {

                double cumulative_sum = 0.0;
                for (const auto idx : cur_level_nodes[node_id].row_indices)
                    cumulative_sum += pool.Target[idx];
                cur_level_nodes[node_id].Value = cumulative_sum
                                           / cur_level_nodes[node_id].row_indices.size();
                left_node.Value = cur_level_nodes[node_id].Value;
                right_node.Value = cur_level_nodes[node_id].Value;
            }

            if (!left_node.is_empty)
                left_node.hists.resize(pool.Features.size());
            if (!right_node.is_empty)
                right_node.hists.resize(pool.Features.size());

            next_level_nodes.emplace_back(left_node);
            next_level_nodes.emplace_back(right_node);
        }

        std::swap(cur_level_nodes, next_level_nodes);
    }

    tree.values.reserve(cur_level_nodes.size());
    for (const auto& node : cur_level_nodes) {
        if (node.is_empty) {
            tree.values.push_back(node.Value);
        } else {
            double cumulative_sum = 0.0;
            for (const auto idx : node.row_indices)
                cumulative_sum += pool.Target[idx];
            tree.values.push_back(cumulative_sum / node.row_indices.size());
        }
    }

    return tree;
}

void TDecisionTree::AddPredict(TPool& pool, float lrate, TTarget& predictions) const {
    for (size_t i = 0; i < pool.Target.size(); ++i) {
        uint32_t predictions_idx = 1;
        for (const auto& it : splits) {
            if (pool.Features[it.first][i] > it.second) {
                predictions_idx <<= 1;
            } else {
                predictions_idx = (predictions_idx << 1) - 1;
            }
        }
        //indexing starts from 0
        --predictions_idx;
        predictions[i] += lrate * values[predictions_idx];
    }

}

void TDecisionTree::ModifyTargetByPredict(TPool& pool, float lrate) const {

    for (size_t i = 0; i < pool.Target.size(); ++i) {
        uint32_t predictions_idx = 1;
        for (const auto& it : splits) {
            if (pool.Features[it.first][i] > it.second) {
                predictions_idx <<= 1;
            } else {
                predictions_idx = (predictions_idx << 1) - 1;
            }
        }
        //indexing starts from 0
        --predictions_idx;
        pool.Target[i] -= lrate * values[predictions_idx];
    }
}