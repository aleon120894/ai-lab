#ifndef AI_LAB_DATA_TYPES_H
#define AI_LAB_DATA_TYPES_H

#include <vector>

/**
 * @brief Represents a single data point (observation) in a dataset.
 * * In supervised learning (like Regression), the features are X and the 
 * cluster_id is ignored. In unsupervised learning (like K-Means), 
 * cluster_id is used to store the assignment, and there is no explicit Y.
 */
struct DataPoint {
    std::vector<double> features; 
    int cluster_id = -1; // -1 means unassigned or not applicable

    // Constructor for easy initialization
    DataPoint(const std::vector<double>& f) : features(f) {}
    DataPoint() = default;
};

#endif // AI_LAB_DATA_TYPES_H
