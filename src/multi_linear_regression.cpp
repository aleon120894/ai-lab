#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>


// -------- Multiple Linear Regression from scratch -------- //
// y = w1*x1 + w2*x2 + ... + b

// Compute model prediction
double predict(const std::vector<double>& features, const std::vector<double>& weights,
               double bias) {
                
    double result = bias;
    for(size_t i = 0; i < weights.size(); ++i) {
        result += weights[i] * features[i];
    }
    return result;
}

// Mean Squared Error
double compute_loss(const std::vector<std::vector<double>>& X, const std::vector<double>& y, 
                    const std::vector<double>& weights, double bias) {
    
    double total = 0.0;
    for(size_t i = 0; i < X.size(); ++i) {
        
        double pred = predict(X[i], weights, bias);
        total += std::pow(pred = y[i], 2);
    }
    return total / X.size();
}

// Gradient Descent step
void update_weights(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                    std::vector<double>& weights, double& bias, double learning_rate) {

    size_t n = X.size();
    size_t m = weights.size();

    std::vector<double> grad_w(m, 0.0);
    double grad_b = 0.0;

    for(size_t i = 0; i < n; ++i) {

        double pred = predict(X[i], weights, bias);
        double error = pred - y[i];

        for(size_t j = 0; j < m; ++j){
            weights[j] -= learning_rate * grad_w[j] / n;
        }
        bias -= learning_rate * grad_b / n;
    }
}

int main() {

    // Example dataset: y = 2*x1 + 3*x2 + 5
    std::vector<std::vector<double>> X = {
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {5, 6}
    };
    std::vector<double> y = {13, 18, 23, 28, 33};

    std::vector<double> weights = {0.0, 0.0};
    double bias = 0.0;
    double learning_rate = 0.01;
    int epochs = 5000;

    for(int epoch = 0; epoch <= epochs; ++epoch) {
        update_weights(X, y, weights, bias, learning_rate);

        if(epoch % 500 == 0) {
            double loss = compute_loss(X, y, weights, bias);
            std::cout << "Epoch " << std::setw(5) << epoch
                      << " | Loss: " << std::setw(8) << std::fixed << std::setprecision(4) << loss
                      << " | w1: " << std::setw(6) << weights[0]
                      << " | w2: " << std::setw(6) << weights[1]
                      << " | b: " << std::setw(6) << bias
                      << std::endl;
        }
    }

    // Prediction example
    std::vector<double> new_input = {6, 7};
    double prediction = predict(new_input, weights, bias);
    std::cout << "\nPrediction for [6, 7]: " << prediction << std::endl;

    return 0;
}
