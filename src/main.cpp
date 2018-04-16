#include "lib/args.hxx"
#include "lib/csv.h"
#include "modes/train.h"

#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
    //аналогично LightGBM у нас будут функции LoadData, Train и Predict

    args::ArgumentParser parser("mini Gradient Boosting utility");
    args::Group commands(parser, "commands");
    args::Command fit(commands, "fit", "Builds model based on provided learning dataset and writes it in output file");
    args::Command predict(commands, "predict", "makes predictions with provided model and test dataset, writes it in output file");
    args::Group arguments(parser, "arguments", args::Group::Validators::DontCare, args::Options::Global);
    args::Positional<std::string> input_file(arguments, "input file", "input for train/predict");
    args::ValueFlag<std::string> column_names_file(arguments, "path", "file containing dataset column names",  { "column_names" });
    args::ValueFlag<std::string> output_file(arguments, "path", "output for train/predict", { "output" });
    args::ValueFlag<std::string> model_file(arguments, "path", "model for predict", { "model-path" });
    args::ValueFlag<std::string> target_column(arguments, "column name", "target column name", { "target" });
    args::ValueFlag<int> iterations(arguments, "iterations amount", "number of trees in ensemble", { "iterations" }, 50);
    args::ValueFlag<float> learning_rate(arguments, "learning-rate", "trees regularization", { "learning-rate" }, 0.8);
    args::ValueFlag<int> depth(arguments, "tree depth", "decision tree max depth", { "iterations" }, 6);

    args::HelpFlag h(arguments, "help", "help", { 'h', "help" });
    //args::PositionalList<std::string> pathsList(arguments, "paths", "files to commit");

    try {
        parser.ParseCLI(argc, argv);
        if (fit) {
            TrainMode::Run(args::get(input_file), args::get(iterations),
                           args::get(learning_rate), args::get(depth));
        } else if (predict) {
            std::cout << "Predict";
        }
    }
    catch (args::Help) {
        std::cout << parser;
    }
    catch (args::Error& e) {
        std::cerr << e.what() << std::endl << parser;
        return 1;
    }
    return 0;
}