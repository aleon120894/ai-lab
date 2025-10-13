#ifndef AI_LAB_LOGISTIC_REGRESSION_H
#define AI_LAB_LOGISTIC_REGRESSION_H

#include "data_types.h"
#include <vector>
#include <cmath>

namespace aicpp {

/**
 * @brief Implements Logistic Regression for binary classification using 
 * Gradient Descent.
 * * It uses the Sigmoid function and Cross-Entropy Loss.
 */
class LogisticRegression {
public:
    /**
     * @brief Constructor for Logistic Regression.
     * @param learning_rate Controls the step size during gradient descent.
     * @param max_iters The maximum number of training epochs.
     */
    LogisticRegression(double learning_rate, int max_iters);

    /**
     * @brief Trains the Logistic Regression model.
     * * @param data The dataset to train on. Assumes the last element of features[i] 
     * is the target label (0 or 1).
     */
    void train(std::vector<DataPoint>& data);

    /**
     * @brief Predicts the probability of the positive class (1) for a given feature set.
     * * @param features The feature vector of the input.
     * @return double The predicted probability (0 to 1).
     */
    double predict_proba(const std::vector<double>& features) const;

    /**
     * @brief Predicts the class (0 or 1) based on a threshold (default 0.5).
     * * @param features The feature vector of the input.
     * @return int The predicted class (0 or 1).
     */
    int predict(const std::vector<double>& features) const;

private:
    double learning_rate_;
    int max_iters_;
    std::vector<double> weights_;
    double bias_;
    size_t num_features_;

    /**
     * @brief The Sigmoid activation function: 1 / (1 + e^(-z)).
     */
    double sigmoid(double z) const;
};

} // namespace aicpp

#endif // AI_LAB_LOGISTIC_REGRESSION_H

