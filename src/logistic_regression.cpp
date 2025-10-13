#include "logistic_regression.h"
#include <iostream>
#include <random>
#include <numeric>
#include <stdexcept>
#include <iomanip>


namespace aicpp {

// --- Private Helpers ---

double LogisticRegression::sigmoid(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

// --- Private Helpers ---

LogisticRegression::LogisticRegression(double learning_rate, int max_iters)
    : learning_rate_(learning_rate), max_iters_(max_iters), bias_(0.0), num_features_(0) {
}

void LogisticRegression::train(std::vector<DataPoint>& data) {
    if (data.empty()) {
        std::cerr << "Warning: Cannot train on an empty dataset." << std::endl;
        return;
    }

    // Assume the last element of the features vector contains the target (Y) and must be separated.
    // The actual features are the rest of the vector.
    num_features_ = data[0].features.size() - 1; 
    if (num_features_ <= 0) {
        throw std::runtime_error("Dataset must contain at least one feature and one target label.");
    }
    
    // Initialize weights and bias to small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    // Using a normal distribution with mean 0 and small standard deviation (0.01) for stable initialization
    std::normal_distribution<> d(0.0, 0.01); 

    weights_.resize(num_features_);
    for (size_t i = 0; i < num_features_; ++i) {
        weights_[i] = d(gen);
    }
    bias_ = d(gen);

    std::cout << "Starting Logistic Regression training (" 
              << num_features_ << " features, " << max_iters_ << " epochs)..." << std::endl;

    // Gradient Descent Loop
    for (int epoch = 0; epoch < max_iters_; ++epoch) {
        // Gradients initialized to zero for the batch
        std::vector<double> dw(num_features_, 0.0);
        double db = 0.0;
        double total_loss = 0.0;

        for (const auto& point : data) {
            // Get target Y (assumed to be the last element)
            double y_true = point.features.back(); 

            // Calculate linear combination (Z = W * X + B) using first num_features_
            double z = bias_;
            for (size_t i = 0; i < num_features_; ++i) {
                z += point.features[i] * weights_[i];
            }

            // Apply sigmoid to get predicted probability (A)
            double y_pred_proba = sigmoid(z);
            
            // Calculate error (A - Y) for gradients
            double error = y_pred_proba - y_true;

            // Update gradients (dw, db) using the average over the batch
            for (size_t i = 0; i < num_features_; ++i) {
                dw[i] += error * point.features[i];
            }
            db += error;

            // Calculate Cross-Entropy Loss for monitoring: -[Y*log(A) + (1-Y)*log(1-A)]
            total_loss += -1.0 * (y_true * std::log(y_pred_proba) + (1.0 - y_true) * std::log(1.0 - y_pred_proba));
        }

        // Calculate average loss and gradients
        double avg_loss = total_loss / data.size();
        for (size_t i = 0; i < num_features_; ++i) {
            dw[i] /= data.size();
        }
        db /= data.size();
        
        // Update weights and bias
        for (size_t i = 0; i < num_features_; ++i) {
            weights_[i] -= learning_rate_ * dw[i];
        }
        bias_ -= learning_rate_ * db;

        // Log progress
        if (epoch % (max_iters_ / 10) == 0 || epoch == max_iters_ - 1) {
            std::cout << "Epoch " << std::setw(4) << std::left << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(5) << avg_loss 
                      << " | Bias: " << std::setprecision(3) << bias_ << std::endl;
        }
    }
    std::cout << "Logistic Regression training finished." << std::endl;
}

double LogisticRegression::predict_proba(const std::vector<double>& features) const {
    if (features.size() != num_features_) {
        throw std::runtime_error("Prediction failed: Feature count mismatch.");
    }

    double z = bias_;
    for (size_t i = 0; i < num_features_; ++i) {
        z += features[i] * weights_[i];
    }
    return sigmoid(z);
}

int LogisticRegression::predict(const std::vector<double>& features) const {
    double probability = predict_proba(features);
    // Simple thresholding: if probability >= 0.5, predict 1, otherwise 0.
    return (probability >= 0.5) ? 1 : 0; 
}

} // namespace aicpp
