#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "csv_parser.h"
#include "data_preprocessor.h"


// Include the new K-Means components
#include "data_types.h"
#include "k_means_clusterer.h"


// Forward declaration for a simple Linear Regression demo (keeping the original structure)
void run_linear_regression_demo();

/**
 * @brief Runs a demonstration of the K-Means Clustering algorithm.
 * * Uses a synthetic 2D dataset to easily visualize (mentally) the clusters.
 */
void run_k_means_demo() {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "           DEMO: K-MEANS CLUSTERING (K=3)              " << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    // 1. Synthetic 2D Dataset (Features: x, y)
    std::vector<DataPoint> dataset = {
        // Cluster 1 (Low values)
        {{1.0, 1.2}}, {{1.5, 1.0}}, {{1.1, 1.5}}, {{2.0, 1.9}}, 
        
        // Cluster 2 (Mid values)
        {{5.0, 5.2}}, {{5.5, 5.0}}, {{4.9, 5.5}}, {{6.0, 5.9}}, 
        
        // Cluster 3 (High values)
        {{9.0, 9.2}}, {{8.5, 9.0}}, {{9.1, 9.5}}, {{10.0, 10.1}}, 
        {{9.5, 8.8}}
    };

    // 2. Initialize and Train the K-Means Clusterer
    const int K = 3;
    const int MAX_ITERS = 100;
    KMeansClusterer clusterer(K, MAX_ITERS);

    clusterer.train(dataset);

    // 3. Print Results
    
    // Display Final Centroids
    std::cout << "\n--- Final Centroids ---" << std::endl;
    const auto& centroids = clusterer.get_centroids();
    for (size_t i = 0; i < centroids.size(); ++i) {
        std::cout << "Centroid " << i << ": (" << centroids[i][0] << ", " << centroids[i][1] << ")" << std::endl;
    }

    // Display Cluster Assignments for the first few points
    std::cout << "\n--- Sample Data Point Assignments ---" << std::endl;
    for (size_t i = 0; i < std::min((size_t)5, dataset.size()); ++i) {
        std::cout << "Point (" << dataset[i].features[0] << ", " << dataset[i].features[1] 
                  << ") assigned to Cluster " << dataset[i].cluster_id << std::endl;
    }
    std::cout << "..." << std::endl;
    for (size_t i = dataset.size() - 5; i < dataset.size(); ++i) {
        std::cout << "Point (" << dataset[i].features[0] << ", " << dataset[i].features[1] 
                  << ") assigned to Cluster " << dataset[i].cluster_id << std::endl;
    }
}


int main() {
    // 1. Run the existing Linear Regression demo
    // run_linear_regression_demo(); 
    
    // 2. Run the new K-Means Clustering demo
    run_k_means_demo();
    // run_logistic_regression_demo();

    return 0;
}


// Mock implementation for the original project's demo
void run_linear_regression_demo() {
    std::cout << "=======================================================" << std::endl;
    std::cout << "      DEMO: LINEAR REGRESSION (GRADIENT DESCENT)       " << std::endl;
    std::cout << "=======================================================" << std::endl;
    // NOTE: Replace this with your actual Linear Regression logic from the original main.cpp
    std::cout << "Running existing Linear Regression model..." << std::endl;
    std::cout << "Epoch 200 | Loss: 310.5 | w: 4.98 b: -50.0" << std::endl;
    std::cout << "Prediction for x=55: ~225" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
}

void run_logistic_regression_demo() {
    std::cout << "=======================================================" << std::endl;
    std::cout << "      DEMO: LOGISTIC REGRESSION                        " << std::endl;
    std::cout << "=======================================================" << std::endl;
}
