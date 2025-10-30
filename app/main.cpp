#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include "core/data_types.h"

// --- K-Means ---
#include "models/clustering/k_means_clusterer.h"

// --- Logistic Regression ---
#include "models/linear/logistic_regression.h"

// --- Multi-Linear Regression (functions from multi_linear_regression.cpp) ---
double mlr_predict(const std::vector<double>& features, const std::vector<double>& weights, double bias) {
    double result = bias;
    for(size_t i = 0; i < weights.size(); ++i) result += weights[i] * features[i];
    return result;
}

double mlr_compute_loss(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                        const std::vector<double>& weights, double bias) {
    double total = 0.0;
    for(size_t i = 0; i < X.size(); ++i) {
        double pred = mlr_predict(X[i], weights, bias);
        total += std::pow(pred - y[i], 2);
    }
    return total / X.size();
}

void mlr_update_weights(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                        std::vector<double>& weights, double& bias, double lr) {
    size_t n = X.size();
    size_t m = weights.size();
    std::vector<double> grad_w(m, 0.0);
    double grad_b = 0.0;

    for(size_t i = 0; i < n; ++i) {
        double pred = mlr_predict(X[i], weights, bias);
        double error = pred - y[i];
        for(size_t j = 0; j < m; ++j) grad_w[j] += error * X[i][j];
        grad_b += error;
    }
    for(size_t j = 0; j < m; ++j) weights[j] -= lr * grad_w[j] / n;
    bias -= lr * grad_b / n;
}

// --- Demos ---
void run_k_means_demo() {
    std::cout << "\n=== K-MEANS CLUSTERING DEMO ===" << std::endl;

    std::vector<DataPoint> dataset = {
        {{1.0, 1.2}}, {{1.5, 1.0}}, {{1.1, 1.5}}, {{2.0, 1.9}},
        {{5.0, 5.2}}, {{5.5, 5.0}}, {{4.9, 5.5}}, {{6.0, 5.9}},
        {{9.0, 9.2}}, {{8.5, 9.0}}, {{9.1, 9.5}}, {{10.0, 10.1}}, {{9.5, 8.8}}
    };

    KMeansClusterer clusterer(3, 100);
    clusterer.train(dataset);

    const auto& centroids = clusterer.get_centroids();
    std::cout << "Centroids:" << std::endl;
    for(size_t i = 0; i < centroids.size(); ++i)
        std::cout << i << ": (" << centroids[i][0] << ", " << centroids[i][1] << ")" << std::endl;
}

void run_multi_linear_regression_demo() {
    std::cout << "\n=== MULTI-LINEAR REGRESSION DEMO ===" << std::endl;

    std::vector<std::vector<double>> X = {{1,2},{2,3},{3,4},{4,5},{5,6}};
    std::vector<double> y = {13, 18, 23, 28, 33};
    std::vector<double> weights(X[0].size(), 0.0);
    double bias = 0.0;
    double lr = 0.01;

    for(int epoch = 0; epoch < 5000; ++epoch) {
        mlr_update_weights(X, y, weights, bias, lr);
        if(epoch % 500 == 0) {
            double loss = mlr_compute_loss(X, y, weights, bias);
            std::cout << "Epoch " << epoch << " | Loss: " << loss
                      << " | w: [" << weights[0] << ", " << weights[1] << "]"
                      << " | b: " << bias << std::endl;
        }
    }

    std::vector<double> new_input = {6,7};
    double pred = mlr_predict(new_input, weights, bias);
    std::cout << "Prediction for [6,7]: " << pred << std::endl;
}

// void run_logistic_regression_demo() {
//     std::cout << "\n=== LOGISTIC REGRESSION DEMO ===" << std::endl;

//     std::vector<aicpp::DataPoint> data = {
//         {{0.0, 0}}, {{1.0, 0}}, {{2.0, 1}}, {{3.0, 1}}
//     };

//     aicpp::LogisticRegression model(0.1, 1000);
//     model.train(data);

//     std::vector<double> new_input = {1.5};
//     int prediction = model.predict(new_input);
//     std::cout << "Prediction for 1.5: " << prediction << std::endl;
// }

int main() {
    run_k_means_demo();
    run_multi_linear_regression_demo();
    // run_logistic_regression_demo();
    return 0;
}


