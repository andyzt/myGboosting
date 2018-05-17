#include "pool.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include "histogram.h"

TRawPool LoadTrainingPool(const std::string& path) {
    TRawPool pool;

    std::ifstream input(path);
    std::string line;

    size_t size = 0;
    size_t featureCount = 0;

    std::vector<bool> CatMask;
    std::vector<std::unordered_map<std::string, size_t>> Hashes;

    while (std::getline(input, line)) {
        if (pool.RawFeatures.empty()) {
            featureCount = size_t(std::count(line.begin(), line.end(), ',')) + 1;
            std::cout << "Detected " << featureCount << " columns" << std::endl;

            for (size_t i = 0; i < featureCount; ++i) {
                pool.RawFeatures.emplace_back();
                pool.RawFeatures.back().reserve(1024);
            }
        }

        std::stringstream stream(line);

        size_t featureId = 0;
        std::string str;

        while (std::getline(stream, str, ',')) {
            float value;

            value = std::stof(str);
            pool.RawFeatures[featureId++].push_back(value);
        }


        if (featureId != featureCount) {
            throw std::length_error("Missing column in line " + std::to_string(size + 1));
        }

        size++;
    }

    if (size == 0) {
        throw std::length_error("Empty file");
    }

    pool.Target = std::move(pool.RawFeatures.back());
    pool.RawFeatures.pop_back();

    return pool;
}

TRawPool LoadTestingPool(const std::string& path, std::vector<std::unordered_map<std::string, size_t>>& hashes) {
    TRawPool pool;

    std::ifstream input(path);
    std::string line;

    size_t size = 0;
    size_t featureCount = 0;

    while (input >> line) {
        if (pool.RawFeatures.empty()) {
            featureCount = size_t(std::count(line.begin(), line.end(), ',')) + 1;
            std::cout << "Detected " << featureCount << " columns" << std::endl;

            for (size_t i = 0; i < featureCount; ++i) {
                pool.RawFeatures.emplace_back();
                pool.RawFeatures.back().reserve(1024);
            }
        }

        std::stringstream stream(line);

        size_t featureId = 0;
        std::string str;

        while (std::getline(stream, str, ',')) {
            float value;


                value = std::stof(str);

            pool.RawFeatures[featureId++].push_back(value);
        }

        if (featureId != featureCount) {
            throw std::length_error("Missing column in line " + std::to_string(size + 1));
        }

        size++;
    }

    if (size == 0) {
        throw std::length_error("Empty file");
    }

    // we have no target in testing dataset
    //pool.Target = std::move(pool.RawFeatures.back());
    pool.RawFeatures.pop_back();

    return pool;
}

TPool ConvertPoolToBinNumbers(const TRawPool& raw, std::vector<std::vector<float>>& bounds) {
    TPool pool;
    pool.Names = std::move(raw.Names);
    pool.Target = std::move(raw.Target);
    pool.Size = raw.RawFeatures[0].size();

    auto rawFeatureCount = raw.RawFeatures.size();
    pool.Features.resize(rawFeatureCount);

    for (size_t rawFeatureId = 0; rawFeatureId < rawFeatureCount; ++rawFeatureId) {
        const auto &rawColumn = raw.RawFeatures[rawFeatureId];
        pool.Features[rawFeatureId].reserve(rawColumn.size());

        for (auto value: rawColumn)
            pool.Features[rawFeatureId].push_back(std::upper_bound(bounds[rawFeatureId].begin(),
                                           bounds[rawFeatureId].end(),
                                           value) - bounds[rawFeatureId].begin());
    }
    pool.FeatureCount = raw.RawFeatures.size();
    return pool;
}