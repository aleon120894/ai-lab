#ifndef AI_LAB_DECISION_TREE_H
#define AI_LAB_DECISION_TREE_H

#include <vector>
#include <memory>
#include "core/data_types.h"

namespace aicpp {

struct TreeNode {
    bool is_leaf = false;          // Leaf flag
    int class_label = -1;          // Final predicted class (for leaves)

    int feature_index = -1;        // Feature index to split by
    double threshold = 0.0;        // Split threshold value

    std::unique_ptr<TreeNode> left;  // Left subtree (feature < threshold)
    std::unique_ptr<TreeNode> right; // Right subtree (feature >= threshold)
};

class DecisionTreeClassifier {
public:

    DecisionTreeClassifier(int max_depth = 3, int min_samples_split = 2);
    
    // Tree output
    void print_tree() const;

    // Train the tree on dataset
    void train(std::vector<DataPoint>& data);

    // Predict class for a single data point
    int predict(const DataPoint& point) const;

    // Batch prediction
    std::vector<int> predict_batch(const std::vector<DataPoint>& points) const;

private:
    std::unique_ptr<TreeNode> root;  // Root of the decision tree
    int MAX_DEPTH;
    int MIN_SAMPLES_SPLIT;

    // Node outout
    void print_node(const TreeNode* node, int depth) const;

    // Recursive tree builder
    std::unique_ptr<TreeNode> build_tree(std::vector<DataPoint>& data, int depth);

    // Gini impurity measure
    double gini_impurity(const std::vector<DataPoint>& data);

    // Weighted cost for split
    double compute_split_cost(const std::vector<DataPoint>& left,
                              const std::vector<DataPoint>& right);
};

} // namespace aicpp

#endif // AI_LAB_DECISION_TREE_H
