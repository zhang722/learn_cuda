// options.h
#pragma once
#include <string>

struct ProgramOptions {
    std::string test_type = "scan";   // -m
    std::string input_type = "random"; // -i
    int array_size = -1;               // -n
    bool use_thrust = false;           // -t
};


ProgramOptions parse_arguments(int argc, char* argv[]);
