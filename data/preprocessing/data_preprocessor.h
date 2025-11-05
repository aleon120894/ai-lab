#pragma once
#include <vector>
#include <string>

class DataPreprocessor {
    
public:
    // Converts string data into numeric feature matrix and label vector
    static void toNumeric(const std::vector<std::vector<std::string>>& raw,
                          std::vector<std::vector<double>>& features,
                          std::vector<double>& labels,
                          int label_col_index);

    // Normalizes features to [0,1] range
    static void normalize(std::vector<std::vector<double>>& features);
};
