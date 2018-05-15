#include "train.h"

#include "algo/binarization.h"
#include "algo/defines.h"
#include "algo/pool.h"
#include "algo/split.h"
#include "algo/tree.h"
#include "algo/model.h"


#include <iostream>
#include <numeric>
#include <sstream>

void TrainMode::Run(const std::string& path, const int iterations, const float lrate, const int depth,
                    const float sample_rate, const int max_bins, const int min_leaf_count,
                    const std::string& output_file) {
    std::cout << "Train" << std::endl;

    std::cout << "Loading " << path << std::endl;


    //TBinarizer binarizer;

    TRawPool pool = LoadTrainingPool(path);

    std::cout << "Done" << std::endl;
    std::cout << "Raw features: " << pool.RawFeatures.size() << std::endl;
    //std::cout << "Binarized features: " << pool.BinarizedFeatureCount << std::endl;
    std::cout << "Size: " << pool.RawFeatures[0].size() << std::endl;


    TModel model;
    model.Fit(std::move(pool), lrate, iterations, sample_rate, depth, min_leaf_count, max_bins);

    std::cout << "Writing to file: " << output_file << std::endl;
    //model.Serialize(output_file, pool);

}
