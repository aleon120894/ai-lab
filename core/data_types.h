#ifndef AI_LAB_DATA_TYPES_H
#define AI_LAB_DATA_TYPES_H

#include <vector>

/**
 * @brief Represents a single data point (observation) in a dataset.
 * * In supervised learning (like Regression), the features are X and the 
 * cluster_id is ignored. In unsupervised learning (like K-Means), 
 * cluster_id is used to store the assignment, and there is no explicit Y.
 */
namespace aicpp {
    struct DataPoint {
    
    std::vector<double> features; 

    // For supervised learning (Decision Tree, Logistic Regression)
    int label = -1;

    // For unsupervised learning (K-Means)
    int cluster_id = -1;


    // Constructor for easy initialization
    DataPoint() = default;
    DataPoint(const std::vector<double>& f, int lbl = -1)
        : features(f), label(lbl) {}
};
} // namespace aicpp

#endif // AI_LAB_DATA_TYPES_H
