#include "predict.h"

#include "algo/binarization.h"
#include "algo/defines.h"
#include "algo/pool.h"
#include "algo/split.h"
#include "algo/tree.h"
#include "algo/model.h"

#include <iostream>
#include <numeric>
#include <fstream>

void PredictMode::Run(const std::string& path, const std::string& model_file, const std::string& output_file) {
    std::cout << "Predict" << std::endl;

    std::cout << "Loading Dataset" << path << std::endl;

    TPool pool;
    TBinarizer binarizer;

    pool = binarizer.Binarize(LoadTrainingPool(path));

    std::cout << "Done" << std::endl;
    std::cout << "Raw features: " << pool.RawFeatureCount << std::endl;
    std::cout << "Binarized features: " << pool.BinarizedFeatureCount << std::endl;
    std::cout << "Size: " << pool.Size << std::endl;

    TModel model(std::move(binarizer));
    model.DeSerialize(model_file);

    auto predictions = model.Predict(std::move(pool));

    std::cout << "Writing to file: " << output_file << std::endl;

    std::ofstream out(output_file);
    for (const auto& val : predictions) {
        //std::cout << val << std::endl;
        out << val << std::endl;
    }

    out.close();
}
