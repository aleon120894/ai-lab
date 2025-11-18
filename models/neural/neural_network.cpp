#include "neural_network.h"
#include <random>
#include <iostream>

namespace aicpp {

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers, double learning_rate)
    : layers_(layers), lr_(learning_rate) {

    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    // allocate weights and biases for layers 1..L-1
    for (size_t l = 1; l < layers_.size(); ++l) {
        int neurons = layers_[l];
        int prev = layers_[l - 1];

        weights_.push_back(std::vector<std::vector<double>>(neurons, std::vector<double>(prev)));
        biases_.push_back(std::vector<double>(neurons, 0.0));

        for (int i = 0; i < neurons; ++i) {
            for (int j = 0; j < prev; ++j) weights_.back()[i][j] = dist(gen);
            biases_.back()[i] = dist(gen);
        }
    }

    // pre-allocate activation structures (mutable, used in const forward)
    activations_.resize(layers_.size());
    zs_.resize(layers_.size()-1); // no z for input layer
}

double NeuralNetwork::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoid_derivative_from_activation(double a) const {
    return a * (1.0 - a); // a = sigmoid(z)
}

double NeuralNetwork::relu(double x) const {
    return x > 0 ? x : 0.0;
}

double NeuralNetwork::relu_derivative(double x) const {
    return x > 0 ? 1.0 : 0.0;
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& x) const {
    // x -> activations_[0]
    activations_[0] = x;

    // for each layer l = 1..L-1
    for (size_t l = 1; l < layers_.size(); ++l) {
        int neurons = layers_[l];
        activations_[l].assign(neurons, 0.0);
        zs_[l-1].assign(neurons, 0.0);

        for (int i = 0; i < neurons; ++i) {
            double z = biases_[l-1][i];
            for (int j = 0; j < layers_[l-1]; ++j) {
                z += weights_[l-1][i][j] * activations_[l-1][j];
            }
            zs_[l-1][i] = z;

            // activation: hidden -> ReLU, output -> sigmoid
            if (l + 1 < layers_.size()) // hidden
                activations_[l][i] = relu(z);
            else // output layer
                activations_[l][i] = sigmoid(z);
        }
    }

    return activations_.back();
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& X,
                          const std::vector<std::vector<double>>& Y,
                          int epochs) {
    const size_t N = X.size();
    if (N == 0) return;

    const int L = static_cast<int>(layers_.size());

    // For batch GD we will accumulate gradients over the whole dataset
    for (int e = 0; e < epochs; ++e) {
        // initialize gradient accumulators with zeros
        std::vector<std::vector<std::vector<double>>> grad_w(weights_.size());
        std::vector<std::vector<double>> grad_b(biases_.size());

        for (size_t l = 0; l < weights_.size(); ++l) {
            grad_w[l].assign(weights_[l].size(), std::vector<double>(weights_[l][0].size(), 0.0));
            grad_b[l].assign(biases_[l].size(), 0.0);
        }

        double total_loss = 0.0;

        // accumulate gradients for each sample
        for (size_t n = 0; n < N; ++n) {
            // forward pass (fills mutable activations_ and zs_)
            forward(X[n]);
            const std::vector<double>& out = activations_.back();

            // compute sample loss (binary: BCE simplified -> but we'll use (out - y)^2/2 for stability)
            for (size_t k = 0; k < out.size(); ++k) {
                double diff = out[k] - Y[n][k];
                total_loss += 0.5 * diff * diff;
            }

            // backprop: compute deltas per layer (from output back to first hidden)
            std::vector<std::vector<double>> delta(L); // delta[l] defined for layer l (same indexing as activations_)
            // output layer delta L-1
            delta[L-1].resize(layers_[L-1]);
            for (int i = 0; i < layers_[L-1]; ++i) {
                double a = activations_[L-1][i];
                double y = Y[n][i];
                // derivative for MSE with sigmoid output: (a - y) * sigmoid'(z)
                double dz = (a - y) * sigmoid_derivative_from_activation(a);
                delta[L-1][i] = dz;
            }

            // hidden layers
            for (int l = L-2; l >= 1; --l) {
                delta[l].resize(layers_[l]);
                for (int i = 0; i < layers_[l]; ++i) {
                    double sum = 0.0;
                    for (int k = 0; k < layers_[l+1]; ++k) {
                        sum += weights_[l][k][i] * delta[l+1][k];
                    }
                    double dz = sum * relu_derivative(zs_[l-1][i]);
                    delta[l][i] = dz;
                }
            }

            // accumulate grads: for layer l = 1..L-1 update grad_w[l-1], grad_b[l-1]
            for (int l = 1; l < L; ++l) {
                for (int i = 0; i < layers_[l]; ++i) {
                    grad_b[l-1][i] += delta[l][i];
                    for (int j = 0; j < layers_[l-1]; ++j) {
                        grad_w[l-1][i][j] += delta[l][i] * activations_[l-1][j];
                    }
                }
            }
        } // end samples

        // average and apply updates
        for (size_t l = 0; l < weights_.size(); ++l) {
            double invN = 1.0 / static_cast<double>(N);

            for (size_t i = 0; i < weights_[l].size(); ++i) {
                biases_[l][i] -= lr_ * (grad_b[l][i] * invN);
                for (size_t j = 0; j < weights_[l][i].size(); ++j) {
                    weights_[l][i][j] -= lr_ * (grad_w[l][i][j] * invN);
                }
            }
        }

        if (e % 500 == 0) {
            std::cout << "Epoch " << e << " | Loss: " << (total_loss / static_cast<double>(N)) << "\n";
        }
    } // epochs
}

std::vector<double> NeuralNetwork::predict_proba(const std::vector<double>& x) const {
    return forward(x);
}

int NeuralNetwork::predict_label(const std::vector<double>& x) const {
    auto p = predict_proba(x);
    return (p.size() > 0 && p[0] >= 0.5) ? 1 : 0;
}

} // namespace aicpp
