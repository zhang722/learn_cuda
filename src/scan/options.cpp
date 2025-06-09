// options.cpp
#include "options.h"
#include <cxxopts.hpp>
#include <iostream>
#include <stdexcept>

ProgramOptions parse_arguments(int argc, char* argv[]) {
    ProgramOptions opts;

    try {
        cxxopts::Options options("cudaScan", "CUDA scan tool");

        options.add_options()
            ("m,test", "Test type", cxxopts::value<std::string>()->default_value("scan"))
            ("i,input", "Input type", cxxopts::value<std::string>()->default_value("random"))
            ("n,arraysize", "Number of elements", cxxopts::value<int>())
            ("t,thrust", "Use thrust", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print usage");

        auto result = options.parse(argc, argv);

        if (result.count("?") || result.count("help")) {
            std::cout << options.help() << std::endl;
            std::exit(0);
        }

        if (!result.count("arraysize")) {
            throw std::runtime_error("Missing required option: -n / --arraysize");
        }

        opts.test_type = result["test"].as<std::string>();
        opts.input_type = result["input"].as<std::string>();
        opts.array_size = result["arraysize"].as<int>();
        opts.use_thrust = result["thrust"].as<bool>();
    }
    catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << std::endl;
        std::exit(1);
    }

    return opts;
}
