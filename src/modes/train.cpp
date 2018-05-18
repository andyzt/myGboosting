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

void TrainMode::Run(const Config& config) {

    if (config.verbose) {
        std::cout << "Train" << std::endl;
        std::cout << "Loading " << config.input_file << std::endl;
    }

    TRawPool pool = LoadPool(config);
    if (config.verbose) {
        std::cout << " Loading ended" << std::endl;

        std::cout << "Raw features: " << pool.RawFeatures.size() << std::endl;
        std::cout << "Size: " << pool.RawFeatures[0].size() << std::endl;
    }
    TModel model;
    model.Fit(pool, config);

    if (config.verbose)
        std::cout << "Writing to file: " << config.output_file << std::endl;
    model.Serialize(config.output_file);

}
