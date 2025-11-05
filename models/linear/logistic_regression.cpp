#include "logistic_regression.h"
#include <iostream>
#include <random>
#include <numeric>
#include <stdexcept>
#include <iomanip>
#include <cmath>


namespace aicpp {

// --- Private Helper ---
double LogisticRegression::sigmoid(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

// --- Constructor ---
LogisticRegression::LogisticRegression(double learning_rate, int max_iters)
    : weights_(), bias_(0.0), learning_rate_(learning_rate), max_iters_(max_iters), num_features_(0) {}

// --- Train method ---
void LogisticRegression::train(std::vector<DataPoint>& data) {

    if (data.empty()) {
        std::cerr << "Warning: Cannot train on empty dataset." << std::endl;
        return;
    }

    num_features_ = data[0].features.size() - 1;
    if (num_features_ <= 0) {
        throw std::runtime_error("Dataset must have at least one feature and one target.");
    }

    // Weights initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 0.01);

    weights_.resize(num_features_);
    for (size_t i = 0; i < num_features_; ++i) weights_[i] = d(gen);
    bias_ = d(gen);

    std::cout << "Starting Logistic Regression training (" 
              << num_features_ << " features, " << max_iters_ << " epochs)..." << std::endl;

    for (int epoch = 0; epoch < max_iters_; ++epoch) {
        std::vector<double> dw(num_features_, 0.0);
        double db = 0.0;
        double total_loss = 0.0;

        for (auto& point : data) {
            double y_true = point.features.back();

            double z = bias_;
            for (size_t i = 0; i < num_features_; ++i) z += point.features[i] * weights_[i];

            double y_pred = sigmoid(z);
            double error = y_pred - y_true;

            for (size_t i = 0; i < num_features_; ++i) dw[i] += error * point.features[i];
            db += error;

            total_loss += -(y_true * std::log(y_pred) + (1.0 - y_true) * std::log(1.0 - y_pred));
        }

        for (size_t i = 0; i < num_features_; ++i) dw[i] /= data.size();
        db /= data.size();
        double avg_loss = total_loss / data.size();

        for (size_t i = 0; i < num_features_; ++i) weights_[i] -= learning_rate_ * dw[i];
        bias_ -= learning_rate_ * db;

        if (epoch % (max_iters_ / 10) == 0 || epoch == max_iters_ - 1) {
            std::cout << "Epoch " << std::setw(4) << std::left << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(5) << avg_loss
                      << " | Bias: " << std::setprecision(3) << bias_ << std::endl;
        }
    }

    std::cout << "Logistic Regression training finished." << std::endl;
}

// --- Predict probability ---
double LogisticRegression::predict_proba(const std::vector<double>& features) const {
    if (features.size() != num_features_) throw std::runtime_error("Feature size mismatch.");

    double z = bias_;
    for (size_t i = 0; i < num_features_; ++i) z += features[i] * weights_[i];
    return sigmoid(z);
}

// --- Predict class ---
int LogisticRegression::predict(const std::vector<double>& features) const {
    return (predict_proba(features) >= 0.5) ? 1 : 0;
}

} // namespace aicpp
