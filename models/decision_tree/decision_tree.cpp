#include "models/decision_tree/decision_tree.h"
#include <algorithm>
#include <iostream>
#include <limits>

namespace aicpp {

DecisionTreeClassifier::DecisionTreeClassifier(int max_depth, int min_samples_split)
    : MAX_DEPTH(max_depth), MIN_SAMPLES_SPLIT(min_samples_split) {}

void DecisionTreeClassifier::train(std::vector<DataPoint>& data) {
    root = build_tree(data, 0);
}

int DecisionTreeClassifier::predict(const DataPoint& point) const {
    const TreeNode* node = root.get();
    while (!node->is_leaf) {
        if (point.features[node->feature_index] < node->threshold)
            node = node->left.get();
        else
            node = node->right.get();
    }
    return node->class_label;
}

std::unique_ptr<TreeNode> DecisionTreeClassifier::build_tree(std::vector<DataPoint>& data, int depth) {
    auto node = std::make_unique<TreeNode>();

    // --- Умова зупинки ---
    if (depth >= MAX_DEPTH || data.size() < MIN_SAMPLES_SPLIT) {
        int count0 = 0, count1 = 0;
        for (auto& d : data) (d.label == 0) ? count0++ : count1++;
        node->is_leaf = true;
        node->class_label = (count1 > count0) ? 1 : 0;
        return node;
    }

    double best_gini = std::numeric_limits<double>::max();
    int best_feature = -1;
    double best_threshold = 0.0;
    std::vector<DataPoint> best_left, best_right;

    int num_features = data[0].features.size();

    for (int feature = 0; feature < num_features; ++feature) {
        for (auto& d : data) {
            double threshold = d.features[feature];

            std::vector<DataPoint> left, right;
            for (auto& sample : data) {
                if (sample.features[feature] < threshold)
                    left.push_back(sample);
                else
                    right.push_back(sample);
            }

            if (left.empty() || right.empty()) continue;

            double cost = compute_split_cost(left, right);
            if (cost < best_gini) {
                best_gini = cost;
                best_feature = feature;
                best_threshold = threshold;
                best_left = left;
                best_right = right;
            }
        }
    }

    if (best_feature == -1) {
        int count0 = 0, count1 = 0;
        for (auto& d : data) (d.label == 0) ? count0++ : count1++;
        node->is_leaf = true;
        node->class_label = (count1 > count0) ? 1 : 0;
        return node;
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = build_tree(best_left, depth + 1);
    node->right = build_tree(best_right, depth + 1);
    return node;
}

double DecisionTreeClassifier::gini_impurity(const std::vector<DataPoint>& data) {
    if (data.empty()) return 0.0;
    int count0 = 0, count1 = 0;
    for (auto& d : data) (d.label == 0) ? count0++ : count1++;
    double p0 = (double)count0 / data.size();
    double p1 = (double)count1 / data.size();
    return 1.0 - (p0 * p0 + p1 * p1);
}

double DecisionTreeClassifier::compute_split_cost(
    const std::vector<DataPoint>& left,
    const std::vector<DataPoint>& right) 
{
    double total = left.size() + right.size();
    double gini_left = gini_impurity(left);
    double gini_right = gini_impurity(right);
    return (left.size() / total) * gini_left + (right.size() / total) * gini_right;
}

void DecisionTreeClassifier::print_tree() const {
    print_node(root.get(), 0);
}


void DecisionTreeClassifier::print_node(const TreeNode* node, int depth) const {
    if (!node) return;
    for(int i = 0; i < depth; ++i) std::cout << "  ";
    if(node->is_leaf)
        std::cout << "Leaf: class=" << node->class_label << "\n";
    else {
        std::cout << "Node: f" << node->feature_index << " < " << node->threshold << "\n";
        print_node(node->left.get(), depth+1);
        print_node(node->right.get(), depth+1);
    }
}

std::vector<int> DecisionTreeClassifier::predict_batch(const std::vector<DataPoint>& points) const {
    std::vector<int> results;
    results.reserve(points.size());
    for (const auto& p : points) {
        results.push_back(predict(p));
    }
    return results;
}

} // namespace aicpp

