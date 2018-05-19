#pragma once

#include<string>
#include <vector>

// represents a single data column
using TRawFeature = std::vector<float>;

// raw features
using TRawFeatures = std::vector<TRawFeature>;

// binarized form of a feature
using TFeature = std::vector<u_int8_t>;

// several feature columns in a vector
using TFeatures = std::vector<TFeature>;

// target column
using TTarget = std::vector<float>;

// feature names
using TNames = std::vector<std::string>;

// a single case to calculate a prediction for
using TRawFeatureRow = std::vector<float>;

// binarized form
using TFeatureRow = std::vector<char>;

// several rows
using TFeatureRows = std::vector<TFeatureRow>;

struct HistogramBin {
    size_t cumulative_cnt = 0;
    double cumulative_sum = 0;
};

//command line parameters
struct Config {
    std::string mode;
    std::string input_file;
    std::string model_file;
    std::string output_file;
    int iterations;
    float learning_rate;
    int depth;
    float sample_rate;
    int max_bins;
    int min_leaf_count;
    int nthreads;
    char delimiter;
    bool has_header;
    bool has_target;
    int target_column_num;
    bool verbose;
};

class TSplit {
public:
    size_t FeatureId = 0;
    size_t bin_id = 0;
    double gain = 0.0;
};

// histogram
using THistogram = std::vector<HistogramBin>;
