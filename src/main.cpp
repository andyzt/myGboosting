#include "lib/args.hxx"
#include "modes/predict.h"
#include "modes/train.h"
//#include <omp.h>

#include <iostream>
#include <fstream>


int main(int argc, char** argv) {
    args::ArgumentParser parser("mini Gradient Boosting utility");
    args::Group commands(parser, "commands");
    args::Command fit(commands, "fit", "Builds model based on provided learning dataset and writes it in output file");
    args::Command predict(commands, "predict",
                          "makes predictions with provided model and test dataset, writes it in output file");
    args::Group arguments(parser, "arguments", args::Group::Validators::DontCare, args::Options::Global);
    args::Positional<std::string> input_file(arguments, "input file", "input for train/predict");
    args::ValueFlag<std::string> column_names_file(arguments, "path",
                                                   "file containing dataset column names",  { "column_names" });
    args::ValueFlag<std::string> output_file(arguments, "path", "output for train/predict", { "output" });
    args::ValueFlag<std::string> model_file(arguments, "path", "model for predict", { "model-path" });
    args::ValueFlag<std::string> target_column(arguments, "column name", "target column name", { "target" });
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
        //omp_set_num_threads(args::get(nthreads));
        //std::cout << " Max threads: " << omp_get_max_threads() << std::endl;
        if (fit) {
            if (args::get(input_file) == "") {
                std::cout << "input file missing: " + input_file.Name() << std::endl;
                return 1;
            }

            if (args::get(output_file) == "") {
                std::cout << "output file missing: " + output_file.Name() << std::endl;
                return 1;
            }

            TrainMode::Run(args::get(input_file), args::get(iterations), args::get(learning_rate), args::get(depth),
                           args::get(sample_rate), args::get(max_bins), args::get(min_leaf_count),
                           args::get(output_file), args::get(verbose)==1);
        } else if (predict) {
            if (args::get(input_file) == "") {
                std::cout << "input file missing: " + input_file.Name() << std::endl;
                return 1;
            }

            if (args::get(model_file) == "") {
                std::cout << "model file missing: " + model_file.Name() << std::endl;
                return 1;
            }

            if (args::get(output_file) == "") {
                std::cout << "output file missing: " + output_file.Name() << std::endl;
                return 1;
            }

            PredictMode::Run(args::get(input_file), args::get(model_file),
                             args::get(output_file), args::get(verbose)==1);
        }
    }
    catch (const args::Help&) {
        std::cout << parser;
    }
    catch (const args::Error& e) {
        std::cerr << e.what() << std::endl << parser;
        return 1;
    }


    return 0;
}