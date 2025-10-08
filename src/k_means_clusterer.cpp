#include "k_means_clusterer.h"
#include <algorithm>
#include <limits>
#include <random>
#include <iostream>
#include <cmath>


KMeansClusterer::KMeansClusterer(int k, int max_iters) :
    K(k), MAX_ITERATIONS(max_iters) {}

/**
 * @brief Calculates the Euclidean distance between two feature vectors.
 * The formula is: d(p, q) = sqrt(sum((qi - pi)^2))
 */
double KMeansClusterer::euclidean_distance(const std::vector<double>& p1, const std::vector<double>& p2) const {
    if (p1.size() != p2.size()) {
        std::cerr << "Error: Vectors must have the same dimension for distance calculation." << std::endl;
        return std::numeric_limits<double>::max();
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        sum_sq += std::pow(p1[i] - p2[i], 2);
    }
    return std::sqrt(sum_sq);
}

/**
 * @brief Initializes centroids by randomly selecting K data points from the dataset.
 */
 void KMeansClusterer::initialize_centroids(const std::vector<DataPoint>& data) {
    if (data.empty()) return;
    
    // Use a random number generator
    std::random_device rd;
    std::mt19937 g(rd());

    // Generate indices from 0 to data.size() - 1
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Randomly shuffle the indices
    std::shuffle(indices.begin(), indices.end(), g);

    // Clear existing centroids and select the first K unique points
    centroids.clear();
    for (int i = 0; i < K; ++i) {
        if (i < data.size()) {
            centroids.push_back(data[indices[i]].features);
        } else {
            // Should not happen if data size >= K, but good safety check
            std::cerr << "Warning: Cannot initialize K centroids, dataset too small." << std::endl;
            break; 
        }
    }
}

/**
 * @brief Assignment step: Assigns each data point to the closest centroid.
 */
void KMeansClusterer::assign_clusters(std::vector<DataPoint>& data) {
    for (auto& point : data) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster_id = -1;

        // Compare the point to every centroid
        for (size_t i = 0; i < centroids.size(); ++i) {
            double dist = euclidean_distance(point.features, centroids[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster_id = (int)i;
            }
        }
        point.cluster_id = best_cluster_id;
    }
}

/**
 * @brief Update step: Recalculates the centroid positions based on assigned points.
 * @return true if centroids moved (indicating non-convergence), false otherwise.
 */
bool KMeansClusterer::update_centroids(const std::vector<DataPoint>& data) {
    // Stores the sum of feature vectors and the count of points for each cluster
    std::vector<std::vector<double>> new_centroids_sum(K, std::vector<double>(centroids[0].size(), 0.0));
    std::vector<int> cluster_counts(K, 0);

    // 1. Sum up all features for points in each cluster
    for (const auto& point : data) {
        int id = point.cluster_id;
        if (id >= 0 && id < K) {
            cluster_counts[id]++;
            for (size_t i = 0; i < point.features.size(); ++i) {
                new_centroids_sum[id][i] += point.features[i];
            }
        }
    }

    // 2. Calculate the mean (average)
    std::vector<std::vector<double>> old_centroids = centroids;
    bool moved = false;
    double convergence_threshold = 1e-6; 

    for (int i = 0; i < K; ++i) {
        if (cluster_counts[i] > 0) {
            for (size_t j = 0; j < centroids[i].size(); ++j) {
                centroids[i][j] = new_centroids_sum[i][j] / cluster_counts[i];
            }
        }
        
        // 3. Check for movement (convergence)
        if (euclidean_distance(centroids[i], old_centroids[i]) > convergence_threshold) {
            moved = true;
        }
    }

    return moved;
}

/**
 * @brief Main training loop for K-Means.
 */
void KMeansClusterer::train(std::vector<DataPoint>& data) {
    if (data.size() < K || data.empty()) {
        std::cerr << "Error: Dataset size is insufficient for K-Means with K=" << K << std::endl;
        return;
    }
    
    // Step 1: Initialization
    initialize_centroids(data);
    std::cout << "--- K-Means Training Started (K=" << K << ") ---" << std::endl;

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        
        // Step 2: Assignment
        assign_clusters(data);

        // Step 3: Update and Check for Convergence
        bool moved = update_centroids(data);

        std::cout << "Iteration " << iter + 1 << ": Centroids updated." << std::endl;

        if (!moved) {
            std::cout << "K-Means converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
        
        if (iter == MAX_ITERATIONS - 1) {
            std::cout << "K-Means reached max iterations (" << MAX_ITERATIONS << ")." << std::endl;
        }
    }
}

