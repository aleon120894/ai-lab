#pragma once
#include <string>
#include <vector>


class CSVParser {
public:
    // Reads CSV file and returns rows as a vector of string vectors
    static std::vector<std::vector<std::string>> readCSV(const std::string& filename, char delimiter = ',');
};
