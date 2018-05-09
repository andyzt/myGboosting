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

void TrainMode::Run(const std::string& path, const int iterations, const float rate, const int depth,
                    const std::string& output_file) {
    std::cout << "Train" << std::endl;

    std::cout << "Loading " << path << std::endl;

    TPool pool;
    TBinarizer binarizer;

    pool = binarizer.Binarize(LoadTrainingPool(path));

    std::cout << "Done" << std::endl;
    std::cout << "Raw features: " << pool.RawFeatureCount << std::endl;
    std::cout << "Binarized features: " << pool.BinarizedFeatureCount << std::endl;
    std::cout << "Size: " << pool.Size << std::endl;

    TModel model(std::move(binarizer));
    model.Fit(std::move(pool), rate, iterations);

    std::cout << "Writing to file: " << output_file << std::endl;
    model.Serialize(output_file, pool);
}
