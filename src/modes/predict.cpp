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

void PredictMode::Run(const std::string& path,
                      const std::string& model_file,
                      const std::string& output_file,
                      const bool verbose) {
    if (verbose) {
        std::cout << "Predict" << std::endl;
        std::cout << "Loading Dataset" << path << std::endl;
    }
    TRawPool raw_pool = LoadTestingPool(path);

    TModel model;

    model.DeSerialize(model_file);

    if (verbose) {
        std::cout << "Done" << std::endl;
        std::cout << "Raw features: " << raw_pool.RawFeatures.size() << std::endl;
        std::cout << "Size: " << raw_pool.RawFeatures[0].size() << std::endl;
    }

    auto predictions = model.PredictOnTestData(raw_pool);

    std::cout << "Writing predictions to file: " << output_file << std::endl;

    std::ofstream out(output_file);
    for (const auto& val : predictions) {
        out << val << std::endl;
    }

    out.close();
 }
