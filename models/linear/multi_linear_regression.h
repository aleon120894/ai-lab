#pragma once
#include <vector>

class MultiLinearRegression {
    
private:
    std::vector<double> weights_;
    double bias_;
    double learning_rate_;

public:
    MultiLinearRegression(double learning_rate = 0.01);

    double predict(const std::vector<double>& features) const;
    double compute_loss(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y) const;
                        
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& y,
               int epochs);
};
