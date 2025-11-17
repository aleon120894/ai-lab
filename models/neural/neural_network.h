#ifndef AI_LAB_NEURAL_NETWORK_H
#define AI_LAB_NEURAL_NETWORK_H

#include <vector>
#include <cstdlib>
#include <cmath>

namespace aicpp {

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layers, double lr = 0.1);

    std::vector<double> predict(const std::vector<double>& x);
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& Y,
               int epochs);

private:
    double lr;  
    std::vector<int> layers;

    // weights[l][i][j] — weight from neuron j in layer l-1 to neuron i in layer l
    std::vector<std::vector<std::vector<double>>> weights;

    // biases[l][i] — bias for neuron i in layer l
    std::vector<std::vector<double>> biases;

    // activations and z-values for backprop
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> zs;

    double sigmoid(double x);
    double sigmoid_derivative(double x);

    std::vector<double> forward(const std::vector<double>& x);
    void backward(const std::vector<double>& y);
};

} // namespace aicpp

#endif
