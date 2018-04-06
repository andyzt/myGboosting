#include <iostream>

#include "src/csv.h"
#include "src/args.hxx"


int main(int argc, char **argv) {

     //аналогично LightGBM у нас будут функции LoadData, Train и Predict

    args::ArgumentParser parser("mini Gradient Boosting utility");
    args::Group commands(parser, "commands");
    args::Command train(commands, "train", "Builds model based on provided learning dataset and writes it in output file");
    args::Command predict(commands, "predict", "makes predictions with provided model and test dataset, writes it in output file");
    args::Group arguments(parser, "arguments", args::Group::Validators::DontCare, args::Options::Global);
    args::Positional<std::string> input_file(arguments, "input file", "input for train/predict");
    args::ValueFlag<std::string> column_names_file(arguments, "path", "file containing dataset column names", {"column_names"});
    args::ValueFlag<std::string> output_file(arguments, "path", "output for train/predict", {"output"});
    args::ValueFlag<std::string> model_file(arguments, "path", "model for predict", {"model"});
    args::ValueFlag<std::string> target_column(arguments, "column name", "Target column name", {"target"});
    args::HelpFlag h(arguments, "help", "help", {'h', "help"});
    //args::PositionalList<std::string> pathsList(arguments, "paths", "files to commit");

    try
    {
        parser.ParseCLI(argc, argv);
        if (train)
        {
            std::cout << "Train";

            io::CSVReader<2, io::trim_chars<' '>, io::double_quote_escape<',','\"'> > in(args::get(input_file));
            std::vector<std::string> names;
            names.push_back("V1");
            names.push_back(args::get(target_column));

            //пришлось исправить этот метод в библиотеке, чтобы он принимал вектор вместо изменяемого числа аргументов
            in.read_header(io::ignore_extra_column, names);
            std::vector<float> row(31);
            //float target;

            //пришлось исправить этот метод в библиотеке, чтобы он принимал вектор вместо изменяемого числа аргументов
            while(in.read_row(row)) {
                // do stuff with the data
            }

        }
        else if (predict)
        {
            std::cout << "Predict";
        }
/*
        for (auto &&path : pathsList)
        {
            std::cout << ' ' << path;
        }

 */       std::cout << std::endl;
    }
    catch (args::Help)
    {
        std::cout << parser;
    }
    catch (args::Error& e)
    {
        std::cerr << e.what() << std::endl << parser;
        return 1;
    }
    return 0;
}