#include "lib/args.hxx"
#include "modes/predict.h"
#include "modes/train.h"
#include "algo/defines.h"
#include <omp.h>

#include <iostream>
#include <fstream>

Config ParseCommandLineArgs(int argc, char** argv) {
    args::ArgumentParser parser("mini Gradient Boosting utility");
    args::Group commands(parser, "commands");
    args::Command fit(commands, "fit", "Builds model based on provided learning dataset and writes it in output file");
    args::Command predict(commands, "predict",
                          "makes predictions with provided model and test dataset, writes it in output file");
    args::Group arguments(parser, "arguments", args::Group::Validators::DontCare, args::Options::Global);
    args::Positional<std::string> input_file(arguments, "input file", "input for train/predict");

    args::ValueFlag<std::string> output_file(arguments, "output path", "output for train/predict", { "output" });
    args::ValueFlag<std::string> model_file(arguments, "model file path", "model for predict", { "model-path" });
    args::ValueFlag<char> delimiter(arguments, "csv delimiter",
                                           "delimiter used in csv file", { "delimiter" },',');
    args::ValueFlag<bool> has_header(arguments, "csv file header",
                                     "if dataset file has header line", { "has-header" }, false);
    args::ValueFlag<bool> has_target(arguments, "has dataset target col",
                                     "if test dataset file has target column", { "has-target" }, false);
    args::ValueFlag<int> target_column_num(arguments, "column number", "target column number", { "target" }, -1);
    args::ValueFlag<int> iterations(arguments, "iterations amount", "number of trees in ensemble", { "iterations" }, 100);
    args::ValueFlag<float> learning_rate(arguments, "learning-rate", "trees regularization", { "learning-rate" }, 1.0);
    args::ValueFlag<float> sample_rate(arguments, "sample-rate",
                                       "percentage of rows for each tree (0 to 1.0)", { "sample-rate" }, 1.0);
    args::ValueFlag<int> depth(arguments, "tree depth", "decision tree max depth", { "depth" }, 6);
    args::ValueFlag<int> max_bins(arguments, "number of bins", "max number of bins in histogram", { "max_bins" }, 10);
    args::ValueFlag<int> nthreads(arguments, "number of threads", "number of parallel pthreads to run", { "nthreads" }, 1);
    args::ValueFlag<int> min_leaf_count(arguments, "min leaf size",
                                        "min number of samples in leaf node", { "min_leaf_count" }, 1);
    args::ValueFlag<int> verbose(arguments, "verbose level", "extended output", { "verbose" }, 0);
    args::HelpFlag h(arguments, "help", "help", { 'h', "help" });
    //args::PositionalList<std::string> pathsList(arguments, "paths", "files to commit");

    try {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Help&) {
        std::cout << parser;
    }
    catch (const args::Error& e) {
        std::cerr << e.what() << std::endl << parser;
        //return 1;
    }
    Config config;
    if (fit)
        config.mode = "fit";
    if (predict)
        config.mode = "predict";
    config.input_file = args::get(input_file);
    config.iterations = args::get(iterations);
    config.learning_rate = args::get(learning_rate);
    config.depth = args::get(depth);
    config.sample_rate = args::get(sample_rate);
    config.max_bins = args::get(max_bins);
    config.min_leaf_count = args::get(min_leaf_count);
    config.output_file = args::get(output_file);
    config.model_file = args::get(model_file);
    config.nthreads = args::get(nthreads);
    config.has_header = args::get(has_header);
    config.has_target = args::get(has_target);
    config.delimiter = args::get(delimiter);
    config.target_column_num = args::get(target_column_num);
    config.verbose = (args::get(verbose)==1);

    return config;
}

int CheckArguments(const Config& config) {
    if (config.mode == "fit") {
        if (config.input_file == "") {
            std::cout << "input file missing "<< std::endl;
            return 1;
        }

        if (config.output_file == "") {
            std::cout << "output file missing "<< std::endl;
            return 1;
        }

        if (config.max_bins > 255 || config.max_bins < 1) {
            std::cout << "max number of bins in histogram should be between 1 and 255 " << std::endl;
            return 1;
        }

        } else if (config.mode == "predict") {
        if (config.input_file == "") {
            std::cout << "input file missing " << std::endl;
            return 1;
        }

        if (config.model_file == "") {
            std::cout << "model file missing " << std::endl;
            return 1;
        }

        if (config.output_file == "") {
            std::cout << "output file missing " << std::endl;
            return 1;
        }
    }
    return 0;
}


int main(int argc, char** argv) {
    Config config = ParseCommandLineArgs(argc, argv);
    if (CheckArguments(config))
        return -1;

    omp_set_num_threads(config.nthreads);

    if (config.verbose)
        std::cout << " Max threads: " << omp_get_max_threads() << std::endl;
    if (config.mode == "fit") {
        TrainMode::Run(config);
    } else if (config.mode == "predict") {
        PredictMode::Run(config);
    }

    return 0;
}
