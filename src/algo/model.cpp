#include "model.h"
#include <fstream>
#include "histogram.h"
#include <limits>


static float MSE(const TTarget& target, const TTarget& test) {
    float mse = 0.0;

    for (size_t i = 0; i < target.size(); ++i) {
        mse += (target[i] - test[i])*(target[i] - test[i]);
    }

    return mse / target.size();
}

TModel::TModel() {}

void TModel::Fit(TRawPool& raw_pool, const Config& config) {
    LearningRate = config.learning_rate;
    TTarget original_target(raw_pool.Target);

    for (size_t l = 0; l < raw_pool.RawFeatures.size(); ++l) {
        upper_bounds.emplace_back(BuildBinBounds(raw_pool.RawFeatures[l], config.max_bins));
    }

    TPool pool = ConvertPoolToBinNumbers(raw_pool, upper_bounds);

    raw_pool.RawFeatures.clear();
    raw_pool.Target.clear();

    double seconds = 0.0;
    clock_t start = clock();

    for (int iter = 0; iter < config.iterations; ++iter) {
        Trees.emplace_back(TDecisionTree::FitHist(pool, config.depth, config.min_leaf_count,
                                                  config.sample_rate, upper_bounds)
        );
        Trees.back().ModifyTargetByPredict(pool, LearningRate);
        std::cout << "Iteration # " << iter << " finished " << std::endl;
        //std::cout << "MSE = " << MSE(target, Predict(pool)) << std::endl;
    }
    clock_t end = clock();
    seconds = double(end - start) / CLOCKS_PER_SEC;
    if (config.verbose) {
        std::cout << "Total time: " << seconds << std::endl;
    }
    std::cout << "MSE = " << MSE(original_target, Predict(pool)) << std::endl;


}

TTarget TModel::Predict(TPool& pool) const {
    TTarget predictions(pool.Target.size(), 0.0);
    for (const auto& tree : Trees) {
            tree.AddPredict(pool, LearningRate, predictions);
    }

    return predictions;
}

TTarget TModel::PredictOnTestData(const TRawPool& raw_pool) const {
    TPool pool = ConvertPoolToBinNumbers(raw_pool, upper_bounds);
    TTarget predictions(pool.Size, 0.0);

    for (const auto& tree : Trees) {
        tree.AddPredict(pool, LearningRate, predictions);
    }

    return predictions;
}

void TModel::Serialize(const std::string& filename) {

    std::ofstream output_file(filename);
    output_file << LearningRate << std::endl;

    //saving all decision trees
    output_file << Trees.size() << std::endl;
    for (const auto& tree : Trees) {
        output_file << tree.splits.size() << std::endl;
        for (const auto& split : tree.splits) {
            output_file << split.first << ' ';
            output_file << static_cast<int>(split.second) << ' ';
        }
        output_file << std::endl;
        output_file << tree.values.size() << std::endl;
        for (const float value : tree.values) {
            output_file << value << ' ';
        }
        output_file << std::endl;
    }

    //saving all upper bounds
    output_file << upper_bounds.size() << std::endl;

    for (const auto& bound : upper_bounds) {
        output_file << bound.size() << std::endl;
        //last val is +inf
        for (int i = 0; i + 1 < bound.size(); ++i) {
            output_file << bound[i] << ' ';
        }
        output_file << std::endl;
    }
}

void TModel::DeSerialize(const std::string& filename) {
    // Read the existing model.
    std::ifstream input(filename);
    if (!input) {
        std::cout << filename << ": File not found." << std::endl;
        return;
    }

    input >> LearningRate;

    size_t N_trees;
    input >> N_trees;
    for (int i=0; i <  N_trees; ++i) {
        TDecisionTree new_tree;
        size_t N_splits;
        input >> N_splits;
        for (int j=0; j <  N_splits; ++j) {
            int feature;
            int bin;
            input >> feature >> bin;
            new_tree.splits.emplace_back(std::make_pair(feature, static_cast<uint8_t>(bin)));
        }
        size_t N_leafs;
        input >> N_leafs;
        for (int j=0; j <  N_leafs; ++j) {
            float val;
            input >> val;
            new_tree.values.emplace_back(val);
        }
        Trees.emplace_back(new_tree);

    }

    //loading all bounds
    size_t N_bounds;
    input >> N_bounds;

    for (int i=0; i <  N_bounds; ++i) {
        std::vector<float> new_bound;
        size_t N_bound_vals;
        input >> N_bound_vals;
        //last val is +inf
        for (int j=0; j + 1<  N_bound_vals; ++j) {
            float bound_val;
            input >> bound_val;
            new_bound.push_back(bound_val);
        }
        new_bound.push_back(std::numeric_limits<float>::max());
        upper_bounds.emplace_back(new_bound);
    }



}

/*
void TModel::SerializeProtobuf(const std::string& filename) {
    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
   GOOGLE_PROTOBUF_VERIFY_VERSION;

    proto_model::Model my_model;
    my_model.set_lr(LearningRate);

    //saving all decision trees
    for (const auto& tree : Trees) {
        auto serialized_tree = my_model.add_tree();
        for (const auto& split : tree.splits) {
            auto serialized_split = serialized_tree->add_splits();
            serialized_split->set_feature_id(split.first);
            serialized_split->set_bin_id(split.second);
        }
        for (const float value : tree.values) {
            auto serialized_leaf = serialized_tree->add_leaf();
            serialized_leaf->set_value(value);
        }

    }

    //saving all upper bounds
    for (const auto& bound : upper_bounds) {
        auto serialized_bound = my_model.add_feature_bounds();
        for (const auto& bound_val : bound)
            serialized_bound->add_bound_val(bound_val);
    }

    // Write the model to disk.
    std::fstream output(filename, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!my_model.SerializeToOstream(&output)) {
        std::cerr << "Failed to write model." << std::endl;

    }

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

}

void TModel::DeSerializeProtobuf(const std::string& filename) {
    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    proto_model::Model my_model;

    // Read the existing model.
    std::fstream input(filename, std::ios::in | std::ios::binary);
    if (!input) {
        std::cout << filename << ": File not found." << std::endl;
        return;
    } else if (!my_model.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse model." << std::endl;
        return;
    }

    LearningRate = my_model.lr();

    for (int i=0; i <  my_model.tree_size(); ++i) {
        TDecisionTree new_tree;
        for (int j=0; j <  my_model.tree(i).splits_size(); ++j) {
            std::pair<int, u_int8_t> new_split;
            new_split.first = my_model.tree(i).splits(j).feature_id();
            new_split.second = my_model.tree(i).splits(j).bin_id();

            new_tree.splits.emplace_back(new_split);
        }

        for (int j=0; j <  my_model.tree(i).leaf_size(); ++j) {
            new_tree.values.emplace_back(my_model.tree(i).leaf(j).value());
        }
        Trees.emplace_back(new_tree);
    }

    //loading all bounds

    for (int i=0; i <  my_model.feature_bounds_size(); ++i) {
        std::vector<float> new_bound;
        for (int j=0; j <  my_model.feature_bounds(i).bound_val_size(); ++j)
            new_bound.push_back(my_model.feature_bounds(i).bound_val(j));
        upper_bounds.emplace_back(new_bound);
    }

        // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

}
*/
