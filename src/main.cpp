#include <iostream>
#include <vector>
#include <cmath>


// Simple dataset price and square of apartment
std::vector<double> X = {40, 50, 60, 70, 80};
std::vector<double> Y = {150, 200, 250, 300, 350};

// Linear regression y = w * x + b
double w = 0.0, b = 0.0;
double lr = 0.0001; // learning rate
int epochs = 1000;

double mse(const std::vector<double> X, const std::vector<double> Y, double w, double b) {

    double error = 0;
    for(size_t i = 0; i < X.size(); i++) {
        double y_pred = w * X[i] + b;
        error += std::pow(Y[i] - y_pred, 2);
    }
    return error / X.size();
}


int main() {

    for (int epoch = 0; epoch < epochs; epoch++) {
        double dw = 0.0, db = 0.0;

        // Gradient calculation
        for (size_t i = 0; i < X.size(); i++) {
            double y_pred = w * X[i] + b;
            dw += -2 * X[i] * (Y[i] - y_pred);
            db += -2 * (Y[i] - y_pred);
        }

        // Updating parameters
        w -= lr * dw / X.size();
        b -= lr * db / X.size();

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch 
                      << " | Loss: " << mse(X, Y, w, b)
                      << " | w: " << w << " b: " << b << std::endl;
        }
    }

    // Test
    double test_x = 55;
    double pred = w * test_x + b;
    std::cout << "Prediction for x=" << test_x << " is " << pred << std::endl;

    return 0;
}
