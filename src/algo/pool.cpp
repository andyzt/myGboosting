#include "pool.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include "histogram.h"
#include "lib/csv.h"

TRawPool LoadPool(const Config& config) {
    TRawPool pool;

    std::ifstream input(config.input_file);
    std::string line;

    size_t size = 0;
    size_t featureCount = 0;

    std::getline(input, line);
    featureCount = size_t(std::count(line.begin(), line.end(), ',')) + 1;
    //std::cout << "Detected " << featureCount << " columns" << std::endl;

    for (size_t i = 0; i < featureCount; ++i) {
        pool.RawFeatures.emplace_back();
        pool.RawFeatures.back().reserve(1024);
    }

    if (config.delimiter == ',') {
        io::CSVReader<io::trim_chars<' '>, io::double_quote_escape<',', '\"'> > in(config.input_file, featureCount);

        std::vector<std::string> col_names(featureCount);
        if (config.has_header)
            in.read_header(io::ignore_missing_column, col_names);

        std::vector<float> row(featureCount);
        while (in.read_row(row)) {
            for (size_t i = 0; i < row.size(); ++i)
                pool.RawFeatures[i].push_back(row[i]);
            size++;
        }
    } else if (config.delimiter == ';') {
        io::CSVReader<io::trim_chars<' '>, io::double_quote_escape<';', '\"'> > in(config.input_file, featureCount);

        std::vector<std::string> col_names(featureCount);
        if (config.has_header)
            in.read_header(io::ignore_missing_column, col_names);

        std::vector<float> row(featureCount);
        while (in.read_row(row)) {
            for (size_t i = 0; i < row.size(); ++i)
                pool.RawFeatures[i].push_back(row[i]);
            size++;
        }
    }

    if (size == 0) {
        throw std::length_error("Empty file");
    }

    // usually we have no target in testing dataset
    if (config.mode =="fit" || config.has_target) {
        if (config.target_column_num == -1) {
            pool.Target = std::move(pool.RawFeatures.back());
            pool.RawFeatures.pop_back();
        } else {
            pool.Target = pool.RawFeatures[config.target_column_num];
            pool.RawFeatures.erase(pool.RawFeatures.begin() + config.target_column_num);
        }
    }

    return pool;
}

TPool ConvertPoolToBinNumbers(const TRawPool& raw, const std::vector<std::vector<float>>& bounds) {
    TPool pool;
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
    return pool;
}