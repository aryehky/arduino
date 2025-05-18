#include "cli.h"
#include <iostream>

int main(int argc, char** argv) {
    try {
        CLI cli(argc, argv);
        cli.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
