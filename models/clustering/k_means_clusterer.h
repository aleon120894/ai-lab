#ifndef AI_LAB_K_MEANS_CLUSTERER_H
#define AI_LAB_K_MEANS_CLUSTERER_H

#include "core/data_types.h"
#include <vector>
#include <cmath>
#include <numeric>

/**
 * @brief Implements the K-Means Clustering algorithm from scratch.
 */
class KMeansClusterer {
public:
    /**
     * @brief Constructor for KMeansClusterer.
     * @param k The number of clusters (K).
     * @param max_iters The maximum number of iterations before stopping.
     */
    KMeansClusterer(int k, int max_iters);

    /**
     * @brief Trains the K-Means model on the provided data.
     * @param data The dataset to cluster. Cluster assignments (cluster_id) 
     * are updated directly in this vector.
     */
    void train(std::vector<DataPoint>& data);

    /**
     * @brief Returns the final calculated centroid coordinates.
     * @return A vector of vectors, where each inner vector is a centroid.
     */
    const std::vector<std::vector<double>>& get_centroids() const {
        return centroids;
    }

private:
    int K;
    int MAX_ITERATIONS;
    // The K cluster centers (centroids)
    std::vector<std::vector<double>> centroids; 

    /**
     * @brief Calculates the Euclidean distance between two feature vectors.
     */
    double euclidean_distance(const std::vector<double>& p1, const std::vector<double>& p2) const;

    /**
     * @brief Initializes centroids by randomly selecting K data points.
     */
    void initialize_centroids(const std::vector<DataPoint>& data);

    /**
     * @brief Assignment step: Assigns each data point to the closest centroid.
     */
    void assign_clusters(std::vector<DataPoint>& data);

    /**
     * @brief Update step: Recalculates the centroid positions based on assigned points.
     * @return true if centroids moved significantly, false otherwise (convergence).
     */
    bool update_centroids(const std::vector<DataPoint>& data);
};

#endif // AI_LAB_K_MEANS_CLUSTERER_H
