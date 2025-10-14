#include "csv_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<std::vector<std::string>> CSVParser::readCSV(const std::string& filename, char delimiter) {
    
    std::vector<std::vector<std::string>> output;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << "\n";
        return output;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delimiter)) {
            row.push_back(cell);
        }
        output.push_back(row);
    }

    file.close();
    return output;
}
