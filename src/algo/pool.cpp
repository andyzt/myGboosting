#include "pool.h"

#include <iostream>
#include <fstream>
#include <sstream>

void TPool::LoadFromFile(const std::string& path, bool train) {
    std::ifstream input(path);
    std::string line;

    while (input >> line) {
        if (Features.empty()) {
            FeatureCount = size_t(std::count(line.begin(), line.end(), ','));
            std::cout << "Detected " << FeatureCount << " columns" << std::endl;
            for (size_t i = 0; i <= FeatureCount; ++i) {
                Features.emplace_back();
                Features.back().reserve(1024);
            }
        }

        std::stringstream stream(line);

        size_t featureId = 0;
        std::string tmp;

        while (std::getline(stream, tmp, ',')) {
            auto value = std::stof(tmp);
            Features[featureId++].push_back(value);
        }

        if (featureId != FeatureCount + 1) {
            throw std::length_error("Missing column in line " + std::to_string(Size + 1));
        }

        Size++;
    }

    if (Size == 0) {
        throw std::length_error("Empty file");
    }

    if (train) {
        Target = std::move(Features.back());
        Features.pop_back();
        FeatureCount--;
    } else {
        Target.reserve(Size);
    }

}
