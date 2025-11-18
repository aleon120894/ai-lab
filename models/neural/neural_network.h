#ifndef AI_LAB_NEURAL_NETWORK_H
#define AI_LAB_NEURAL_NETWORK_H

#include <vector>
#include <cstdlib>
#include <cmath>

namespace aicpp {

class NeuralNetwork {
public:

    // layers: e.g. {input_dim, hidden1, hidden2, output_dim}
    NeuralNetwork(const std::vector<int>& layers, double learning_rate = 0.1);

    // Train with full-batch gradient descent
    // X: NxD, Y: NxO (for binary classification O=1)
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& Y,
               int epochs);

    // Predict probabilities (sigmoid outputs)
    std::vector<double> predict_proba(const std::vector<double>& x) const;

    // Predict class (0/1) using 0.5 threshold
    int predict_label(const std::vector<double>& x) const;

private:
    std::vector<int> layers_;
    double lr_;

    // weights_[l][i][j] — layer l (1..L-1), neuron i, prev neuron j
    std::vector<std::vector<std::vector<double>>> weights_;
    std::vector<std::vector<double>> biases_;

    // helpers used per-sample during forward/backward
    mutable std::vector<std::vector<double>> activations_; // size = layers_.size()
    mutable std::vector<std::vector<double>> zs_;          // pre-activations (for hidden+out)

    double sigmoid(double x) const;
    double sigmoid_derivative_from_activation(double a) const;
    double relu(double x) const;
    double relu_derivative(double x) const;

    std::vector<double> forward(const std::vector<double>& x) const;
};

} // namespace aicpp

#endif
