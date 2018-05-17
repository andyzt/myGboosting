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

    TRawPool raw_pool = LoadTestingPool(model_file);

    TModel model;

    model.DeSerialize(model_file);


    std::cout << "Done" << std::endl;
    std::cout << "Raw features: " << raw_pool.RawFeatures.size() << std::endl;
    std::cout << "Size: " << raw_pool.Target.size() << std::endl;


    auto predictions = model.PredictOnTestData(raw_pool);

    std::cout << "Writing to file: " << output_file << std::endl;

    std::ofstream out(output_file);
    for (const auto& val : predictions) {
        out << val << std::endl;
    }

    out.close();
 }
