#include "pool.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>

TRawPool LoadTrainingPool(const std::string& path) {
    TRawPool pool;

    std::ifstream input(path);
    std::string line;

    size_t size = 0;
    size_t featureCount = 0;
    bool initialized = false;
    std::vector<bool> CatMask;
    std::vector<std::unordered_map<std::string, size_t>> Hashes;

    while (input >> line) {
        if (pool.RawFeatures.empty()) {
            featureCount = size_t(std::count(line.begin(), line.end(), ',')) + 1;
            std::cout << "Detected " << featureCount << " columns" << std::endl;
            CatMask.resize(featureCount, 0);
            pool.Hashes.resize(featureCount);
            pool.Ranges.resize(featureCount, std::make_pair(std::numeric_limits<float>::max(),
                                                            std::numeric_limits<float>::min()));

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

            if (!initialized) {
                try {
                    value = std::stof(str);
                } catch (...) {
                    CatMask[featureId] = 1;
                }
            }

            if (CatMask[featureId] != 0) {
                auto& hash = pool.Hashes[featureId];
                auto it = hash.find(str);
                if (it == hash.end()) {
                    value = hash.size();
                    hash[str] = size_t(value);
                } else {
                    value = it->second;
                }
            } else {
                value = std::stof(str);
                auto& min = pool.Ranges[featureId].first;
                min = std::min(value, min);
                auto& max = pool.Ranges[featureId].second;
                min = std::max(value, max);
            }

            pool.RawFeatures[featureId++].push_back(value);
        }

        initialized = true;

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
    bool initialized = false;
    std::vector<bool> CatMask;

    while (input >> line) {
        if (pool.RawFeatures.empty()) {
            featureCount = size_t(std::count(line.begin(), line.end(), ',')) + 1;
            std::cout << "Detected " << featureCount << " columns" << std::endl;
            CatMask.resize(featureCount, 0);
            //copy exitsing hashes
            pool.Hashes = std::move(hashes);
            pool.Ranges.resize(featureCount, std::make_pair(std::numeric_limits<float>::max(),
                                                            std::numeric_limits<float>::min()));

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

            if (!initialized) {
                try {
                    value = std::stof(str);
                } catch (...) {
                    CatMask[featureId] = 1;
                }
            }

            if (CatMask[featureId] != 0) {
                auto& hash = pool.Hashes[featureId];
                auto it = hash.find(str);
                if (it == hash.end()) {
                    value = hash.size();
                    hash[str] = size_t(value);
                } else {
                    value = it->second;
                }
            } else {
                value = std::stof(str);
                auto& min = pool.Ranges[featureId].first;
                min = std::min(value, min);
                auto& max = pool.Ranges[featureId].second;
                min = std::max(value, max);
            }

            pool.RawFeatures[featureId++].push_back(value);
        }

        initialized = true;

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

TPool ConvertPoolToBinNumbers(TRawPool&& raw, int max_bins) {
    TPool pool;
    pool.Names = std::move(raw.Names);
    pool.Target = std::move(raw.Target);
    pool.Size = raw.RawFeatures[0].size();

    pool.Hashes = std::move(raw.Hashes);

    auto rawFeatureCount = raw.RawFeatures.size();

    /*
    for (size_t rawFeatureId = 0; rawFeatureId < rawFeatureCount; ++rawFeatureId) {
        TFeatures binarized;
        const auto& rawColumn = raw.RawFeatures[rawFeatureId];
        if (!pool.Hashes[rawFeatureId].empty()) {
            binarized = BinarizeCatFeature(rawColumn, pool.Hashes[rawFeatureId].size());
        } else {
            auto splits = BuildSplits(raw.RawFeatures[rawFeatureId], max_bins);
            binarized = BinarizeFloatFeature(rawColumn, splits);
            Splits.emplace_back(std::move(splits));
        }

        for (auto& column : binarized) {
            pool.Features.emplace_back(std::move(column));
            BinarizedToRaw.push_back(rawFeatureId);
        }
    }

    pool.RawFeatureCount = raw.RawFeatures.size();
    pool.BinarizedFeatureCount = pool.Features.size();

    pool.Rows = SetupTestData(pool);
     */

    return pool;
}