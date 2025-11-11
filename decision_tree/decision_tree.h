#ifndef AI_LAB_DECISION_TREE_H
#define AI_LAB_DECISION_TREE_H

#include <vector>
#include <memory>
#include <limits>
#include "core/data_types.h"

namespace aicpp {

struct TreeNode {
    bool is_leaf = false;
    // int predicted_label = -1; 
    int class_label = -1;

    int feature_index = -1;
    double threshold = 0.0;

    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
};

class DecisionTreeClassifier {
public:
    DecisionTreeClassifier(int max_depth = 3, int min_samples_split = 2);

    void train(std::vector<DataPoint>& data);
    int predict(const DataPoint& point) const;

private:
    std::unique_ptr<TreeNode> root;
    int MAX_DEPTH;
    int MIN_SAMPLES_SPLIT;

    std::unique_ptr<TreeNode> build_tree(std::vector<DataPoint>& data, int depth);
    
    double gini_impurity(const std::vector<DataPoint>& data);
    double compute_split_cost(const std::vector<DataPoint>& left,
                              const std::vector<DataPoint>& right);
};

} // namespace aicpp

#endif // AI_LAB_DECISION_TREE_H
