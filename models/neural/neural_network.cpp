#include "neural_network.h"
#include <random>
#include <iostream>

namespace aicpp {

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers, double lr)
        : layers(layers), lr(lr) {

    std::mt19937 gen(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Allocate weights and biases
    for (size_t l = 1; l < layers.size(); ++l) {
        int neurons = layers[l];
        int prev = layers[l - 1];

        weights.push_back(std::vector<std::vector<double>>(neurons,
                 std::vector<double>(prev)));
        biases.push_back(std::vector<double>(neurons));

        for (int i = 0; i < neurons; i++) {
            for (int j = 0; j < prev; j++)
                weights.back()[i][j] = dist(gen);

            biases.back()[i] = dist(gen);
        }
    }
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& x) {
    activations.clear();
    zs.clear();

    activations.push_back(x);

    for (size_t l = 1; l < layers.size(); ++l) {
        int neurons = layers[l];
        int prev = layers[l - 1];

        std::vector<double> z(neurons);
        std::vector<double> a(neurons);

        for (int i = 0; i < neurons; i++) {
            z[i] = biases[l - 1][i];

            for (int j = 0; j < prev; j++)
                z[i] += weights[l - 1][i][j] * activations.back()[j];

            a[i] = sigmoid(z[i]);
        }

        zs.push_back(z);
        activations.push_back(a);
    }

    return activations.back();
}

void NeuralNetwork::backward(const std::vector<double>& y) {
    int L = layers.size() - 1;

    // delta for output layer
    std::vector<std::vector<double>> delta(L);
    delta[L - 1].resize(layers[L]);

    for (int i = 0; i < layers[L]; i++)
        delta[L - 1][i] =
            (activations[L][i] - y[i]) * sigmoid_derivative(zs[L - 1][i]);

    // Backprop hidden layers
    for (int l = L - 1; l > 0; l--) {
        delta[l - 1].resize(layers[l]);
        for (int i = 0; i < layers[l]; i++) {
            double error = 0;
            for (int k = 0; k < layers[l + 1]; k++)
                error += delta[l][k] * weights[l][k][i];

            delta[l - 1][i] = error * sigmoid_derivative(zs[l - 1][i]);
        }
    }

    // Update weights & biases
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < layers[l + 1]; i++) {
            biases[l][i] -= lr * delta[l][i];

            for (int j = 0; j < layers[l]; j++)
                weights[l][i][j] -= lr * delta[l][i] * activations[l][j];
        }
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& X,
                          const std::vector<std::vector<double>>& Y,
                          int epochs) {

    for (int e = 0; e < epochs; e++) {
        double loss = 0;

        for (size_t i = 0; i < X.size(); i++) {
            auto out = forward(X[i]);
            backward(Y[i]);

            for (int k = 0; k < out.size(); k++)
                loss += 0.5 * (out[k] - Y[i][k]) * (out[k] - Y[i][k]);
        }

        if (e % 1000 == 0)
            std::cout << "Epoch " << e << " | Loss: " << loss / X.size() << "\n";
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& x) {
    return forward(x);
}

} // namespace aicpp
