#include "data_preprocessor.h"
#include <limits>
#include <cmath>
#include <cstdlib>


void DataPreprocessor::toNumeric(const std::vector<std::vector<std::string>>& raw,
                                 std::vector<std::vector<double>>& features,
                                 std::vector<double>& labels,
                                 int label_col_index) {
    int n = raw.size();
    if (n == 0) return;
    int m = raw[0].size();

    // Skip header (row 0)
    for (int i = 1; i < n; ++i) {

        std::vector<double> row;
        for (int j = 0; j < m; ++j) {
            if (j == label_col_index) continue;
            double val = std::stod(raw[i][j]);
            row.push_back(val);
        }
        
        features.push_back(row);
        double label = std::stod(raw[i][label_col_index]);
        labels.push_back(label);
    }
}

void DataPreprocessor::normalize(std::vector<std::vector<double>>& features) {

    if (features.empty()) return;

    int n = features.size();
    int m = features[0].size();
    std::vector<double> minv(m, std::numeric_limits<double>::infinity());
    std::vector<double> maxv(m, -std::numeric_limits<double>::infinity());

    // Compute min and max for each column
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            minv[j] = std::min(minv[j], features[i][j]);
            maxv[j] = std::max(maxv[j], features[i][j]);
        }
    }

    // Normalize to [0,1]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (maxv[j] - minv[j] != 0)
                features[i][j] = (features[i][j] - minv[j]) / (maxv[j] - minv[j]);
            else
                features[i][j] = 0.0;
        }
    }
}
