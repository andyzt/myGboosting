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

    TRawPool pool;
    //TBinarizer binarizer;
    TModel model;

    std::vector<std::vector<float>> splits;
    std::vector<std::unordered_map<std::string, size_t>> hashes;
    //model.DeSerialize(model_file, hashes, splits);

    //pool = binarizer.BinarizeTestData(LoadTestingPool(path, hashes), splits);

    std::cout << "Done" << std::endl;
    std::cout << "Raw features: " << pool.RawFeatures.size() << std::endl;
    //std::cout << "Binarized features: " << pool.BinarizedFeatureCount << std::endl;
    std::cout << "Size: " << pool.Target.size() << std::endl;

    //TModel model(std::move(binarizer));

/*
    auto predictions = model.Predict(pool);

    std::cout << "Writing to file: " << output_file << std::endl;

    std::ofstream out(output_file);
    for (const auto& val : predictions) {
        //std::cout << val << std::endl;
        out << val << std::endl;
    }

    out.close();
*/
 }
