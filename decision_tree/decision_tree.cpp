#include "decision_tree.h"
#include <algorithm>
#include <cmath>

namespace aicpp {

DecisionTreeClassifier::DecisionTreeClassifier(int max_depth, int min_samples_split)
    : MAX_DEPTH(max_depth), MIN_SAMPLES_SPLIT(min_samples_split) {}

void DecisionTreeClassifier::train(std::vector<DataPoint>& data) {
    root = build_tree(data, 0);
}

int DecisionTreeClassifier::predict(const DataPoint& point) const {
    TreeNode* node = root.get();
    while (!node->is_leaf) {
        if (point.label < 0) {
            // features-only inference
        }
        if (point.features[node->feature_index] < node->threshold)
            node = node->left.get();
        else
            node = node->right.get();
    }
    return node->predicted_label;
}

std::unique_ptr<TreeNode> DecisionTreeClassifier::build_tree(std::vector<DataPoint>& data, int depth) {
    auto node = std::make_unique<TreeNode>();

    // Stop conditions TBD in stage 2
    node->is_leaf = true;
    node->predicted_label = data.empty() ? 0 : data[0].label;
    return node;
}

// Gini helper
double DecisionTreeClassifier::gini_impurity(const std::vector<DataPoint>& data) {
    if (data.empty()) return 0.0;

    int count0 = 0;
    int count1 = 0;

    for (const auto& dp : data) {
        if (dp.label == 0) count0++;
        else count1++;
    }

    double p0 = (double)count0 / data.size();
    double p1 = (double)count1 / data.size();

    return 1.0 - (p0*p0 + p1*p1);
}

double DecisionTreeClassifier::compute_split_cost(
        const std::vector<DataPoint>& left,
        const std::vector<DataPoint>& right) {

    double total = left.size() + right.size();
    return (left.size() / total) * gini_impurity(left) +
           (right.size() / total) * gini_impurity(right);
}

} // namespace aicpp
