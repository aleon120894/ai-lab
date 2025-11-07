#ifndef AI_LAB_K_MEANS_CLUSTERER_H
#define AI_LAB_K_MEANS_CLUSTERER_H

#include "core/data_types.h"
#include <vector>
#include <cmath>
#include <numeric>

namespace aicpp {

/**
 * @brief Implements the K-Means Clustering algorithm from scratch.
 */
class KMeansClusterer {

public:
    KMeansClusterer(int k, int max_iters);

    /**
     * @brief Trains the K-Means model on the provided data.
     */
    void train(std::vector<DataPoint>& data);

    /**
     * @brief Returns the final calculated centroid coordinates.
     */
    const std::vector<std::vector<double>>& get_centroids() const {
        return centroids;
    }

private:
    int K;
    int MAX_ITERATIONS;

    std::vector<std::vector<double>> centroids;

    double euclidean_distance(const std::vector<double>& p1,
                              const std::vector<double>& p2) const;

    void initialize_centroids(const std::vector<DataPoint>& data);
    void assign_clusters(std::vector<DataPoint>& data);
    bool update_centroids(const std::vector<DataPoint>& data);
};

} // namespace aicpp

#endif // AI_LAB_K_MEANS_CLUSTERER_H

