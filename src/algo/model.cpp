#include "model.h"

static float MSE(const TTarget& target, const TTarget& test) {
    float mse = 0.0;
    for (size_t i = 0; i < target.size(); ++i) {
        mse += (target[i] - test[i])*(target[i] - test[i]);
    }

    return mse / target.size();
}

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
            pool.Target[i] -= rate*tree.Predict(pool.Rows[i]);
        }

        std::cout << "MSE = " << MSE(target, Predict(pool)) << std::endl;
    }
}

TTarget TModel::Predict(const TPool& pool) const {
    TTarget predictions(pool.Size, 0.0);
    for (const auto& tree : Trees) {
        for (size_t i = 0; i < pool.Size; ++i) {
            predictions[i] += tree.Predict(pool.Rows[i]);
        }
    }

    return predictions;
}

//TTarget TModel::Predict(const TRawPool& raw) const {
//    return TTarget();
//}
