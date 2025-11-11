#include "decision_tree.h"
#include <algorithm>
#include <cmath>

namespace aicpp {

DecisionTreeClassifier::DecisionTreeClassifier(int max_depth, int min_samples_split)
    : MAX_DEPTH(max_depth), MIN_SAMPLES_SPLIT(min_samples_split) {}

double DecisionTreeClassifier::gini_impurity(const std::vector<DataPoint>& data) {
    if (data.empty()) return 0.0;
    int count0 = 0, count1 = 0;
    for (const auto& d : data) {
        (d.label == 0) ? count0++ : count1++;
    }
    double p0 = static_cast<double>(count0) / data.size();
    double p1 = static_cast<double>(count1) / data.size();
    return 1.0 - p0 * p0 - p1 * p1;
}

double DecisionTreeClassifier::compute_split_cost(
        const std::vector<DataPoint>& left,
        const std::vector<DataPoint>& right) {
    double total = left.size() + right.size();
    return (left.size() / total) * gini_impurity(left) +
           (right.size() / total) * gini_impurity(right);
}

std::unique_ptr<TreeNode> DecisionTreeClassifier::build_tree(
        std::vector<DataPoint>& data, int depth) {

    auto node = std::make_unique<TreeNode>();

    // --- Stopping conditions ---
    if (depth >= MAX_DEPTH || data.size() < MIN_SAMPLES_SPLIT ||
        gini_impurity(data) == 0.0) {
        node->is_leaf = true;
        int count1 = 0;
        for (auto& d : data) if (d.label == 1) count1++;
        node->class_label = (count1 > data.size() / 2) ? 1 : 0;
        return node;
    }

    int num_features = data[0].features.size();
    double best_cost = 1e9;
    int best_feature = 0;
    double best_threshold = 0;

    for (int f = 0; f < num_features; f++) {
        for (const auto& d : data) {
            double threshold = d.features[f];
            std::vector<DataPoint> left, right;
            for (const auto& dp : data) {
                (dp.features[f] < threshold) ? left.push_back(dp) : right.push_back(dp);
            }
            if (left.empty() || right.empty()) continue;

            double cost = compute_split_cost(left, right);
            if (cost < best_cost) {
                best_cost = cost;
                best_feature = f;
                best_threshold = threshold;
            }
        }
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;

    std::vector<DataPoint> left, right;
    for (const auto& d : data) {
        (d.features[best_feature] < best_threshold)
            ? left.push_back(d)
            : right.push_back(d);
    }

    node->left = build_tree(left, depth + 1);
    node->right = build_tree(right, depth + 1);

    return node;
}

void DecisionTreeClassifier::train(std::vector<DataPoint>& data) {
    root = build_tree(data, 0);
}

int DecisionTreeClassifier::predict(const DataPoint& point) const {
    TreeNode* node = root.get();
    while (!node->is_leaf) {
        node = (point.features[node->feature_index] < node->threshold)
                ? node->left.get()
                : node->right.get();
    }
    return node->class_label;
}

} // namespace aicpp
