#pragma once
#include <vector>
#include <cstddef> // for size_t
#include "../../core/data_types.h"


namespace aicpp {

class LogisticRegression {
public:
    LogisticRegression(double learning_rate, int max_iters);

    void train(std::vector<DataPoint>& data);

    double predict_proba(const std::vector<double>& features) const;
    int predict(const std::vector<double>& features) const;

private:
    double sigmoid(double z) const;

    std::vector<double> weights_;
    double bias_;
    double learning_rate_;
    
    int max_iters_;
    size_t num_features_; 
};

} // namespace aicpp 
