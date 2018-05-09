#include "model.h"
#include "proto/model.pb.h"
#include <fstream>


static float MSE(const TTarget& target, const TTarget& test) {
    float mse = 0.0;
    for (size_t i = 0; i < target.size(); ++i) {
        mse += (target[i] - test[i])*(target[i] - test[i]);
    }

    return mse / target.size();
}

TModel::TModel() {}

TModel::TModel(TBinarizer&& binarizer)
    : Binarizer(std::forward<TBinarizer>(binarizer)) {

}

void TModel::Fit(TPool&& pool, float rate, float iterations) {
    LearningRate = rate;
    TTarget target(pool.Target);

    for (int iter = 0; iter < iterations; ++iter) {
        Trees.push_back(TDecisionTree::Fit(pool, 6, 10, false));

        const auto& tree = Trees.back();

        //replacing our target by gradient of current step
        for (size_t i = 0; i < pool.Size; ++i) {
            pool.Target[i] -= LearningRate*tree.Predict(pool.Rows[i]);
        }

        std::cout << "MSE = " << MSE(target, Predict(pool)) << std::endl;
    }
}

TTarget TModel::Predict(const TPool& pool) const {
    TTarget predictions(pool.Size, 0.0);
    for (const auto& tree : Trees) {
        for (size_t i = 0; i < pool.Size; ++i) {
            predictions[i] += LearningRate*tree.Predict(pool.Rows[i]);
        }
    }

    return predictions;
}

//TTarget TModel::Predict(const TRawPool& raw) const {
//    return TTarget();
//}

void TModel::Serialize(const std::string& filename, const TPool& pool) {
    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    proto_model::Model my_model;
    my_model.set_lr(LearningRate);

    //saving all decision trees
    for (const auto& tree : Trees) {
        auto serialized_tree = my_model.add_tree();
        for (const auto& node : tree.Nodes) {
            auto serialized_node = serialized_tree->add_node();
            serialized_node->set_featureid(node.FeatureId);
            serialized_node->set_left(node.Left);
            serialized_node->set_right(node.Right);
            serialized_node->set_leaf(node.Leaf);
            serialized_node->set_value(node.Value);
        }
    }

    //saving all splits
    for (const auto& split : Binarizer.GetSplits()) {
        auto serialized_split = my_model.add_splits();
        for (const auto& split_val : split)
            serialized_split->add_split_val(split_val);
    }

    //saving all hashes
    for (const auto& hash : pool.Hashes) {
        auto serialized_hash = my_model.add_hashes();
        for (auto it : hash)
            (*serialized_hash->mutable_hash())[it.first] = it.second;

    }

    // Write the model to disk.
    std::fstream output(filename, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!my_model.SerializeToOstream(&output)) {
        std::cerr << "Failed to write model." << std::endl;

    }

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

}

void TModel::DeSerialize(const std::string& filename,
                         std::vector<std::unordered_map<std::string, size_t>>& hashes,
                         std::vector<std::vector<float>>& splits) {
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
        for (int j=0; j <  my_model.tree(i).node_size(); ++j) {
            TDecisionTreeNode new_node;
            new_node.FeatureId = my_model.tree(i).node(j).featureid();
            new_node.Left = my_model.tree(i).node(j).left();
            new_node.Right = my_model.tree(i).node(j).right();
            new_node.Leaf = my_model.tree(i).node(j).leaf();
            new_node.Value = my_model.tree(i).node(j).value();

            new_tree.Nodes.emplace_back(new_node);
        }
        Trees.emplace_back(new_tree);
    }

    //loading all splits

    for (int i=0; i <  my_model.splits_size(); ++i) {
        std::vector<float> new_split;
        for (int j=0; j <  my_model.splits(i).split_val_size(); ++j)
            new_split.push_back(my_model.splits(i).split_val(j));
        splits.emplace_back(new_split);
    }

    //loading all hashes
    for (int i=0; i <  my_model.hashes_size(); ++i) {
        std::unordered_map<std::string, size_t> new_hash;
        for (auto it : my_model.hashes(i).hash())
            new_hash[it.first] = it.second;
        hashes.emplace_back(new_hash);
    }


        // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

}