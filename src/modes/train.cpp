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
                    const std::string& output_file, const bool verbose) {

    if (verbose) {
        std::cout << "Train" << std::endl;
        std::cout << "Loading " << path << std::endl;
    }

    TRawPool pool = LoadTrainingPool(path);
    if (verbose) {
        std::cout << " Loading ended" << std::endl;

        std::cout << "Raw features: " << pool.RawFeatures.size() << std::endl;
        std::cout << "Size: " << pool.RawFeatures[0].size() << std::endl;
    }
    TModel model;
    model.Fit(pool, lrate, iterations, sample_rate, depth, min_leaf_count, max_bins);

    if (verbose)
        std::cout << "Writing to file: " << output_file << std::endl;
    model.Serialize(output_file);

}
