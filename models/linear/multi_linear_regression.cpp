#include "multi_linear_regression.h"
#include <cmath>
#include <iostream>


MultiLinearRegression::MultiLinearRegression(double learning_rate)
    : learning_rate_(learning_rate), bias_(0.0) {}

double MultiLinearRegression::predict(const std::vector<double>& features) const {

    double r = bias_;
    for (size_t i = 0; i < weights_.size(); i++)
        r += weights_[i] * features[i];
    return r;
}

double MultiLinearRegression::compute_loss(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y) const
{
    double total = 0;
    
    for (size_t i = 0; i < X.size(); i++)
        total += std::pow(predict(X[i]) - y[i], 2);
    return total / X.size();
}

void MultiLinearRegression::train(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    int epochs)
{
    size_t n = X.size(), m = X[0].size();
    weights_.assign(m, 0.0);

    for (int e = 0; e <= epochs; e++) {

        std::vector<double> grad_w(m, 0.0);
        double grad_b = 0;

        for (size_t i = 0; i < n; i++) {
            
            double pred = predict(X[i]);
            double err = pred - y[i];

            for (size_t j = 0; j < m; j++) grad_w[j] += err * X[i][j];
            grad_b += err;
        }

        for (size_t j = 0; j < m; j++)
            weights_[j] -= learning_rate_ * grad_w[j] / n;
        bias_ -= learning_rate_ * grad_b / n;

        if (e % 500 == 0)
            std::cout << "Epoch " << e
                      << " | Loss=" << compute_loss(X, y)
                      << " | b=" << bias_ << "\n";
    }
}

