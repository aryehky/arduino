#include "color_spaces.h"
#include <algorithm>
#include <random>
#include <map>
#include <queue>

namespace preprocessing {

std::vector<double> ColorSpaces::quantizeColors(const std::vector<double>& image,
                                              int width,
                                              int height,
                                              int num_colors) {
    if (!validateImageDimensions(image, width * 3, height)) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    if (num_colors < 2 || num_colors > 256) {
        throw std::invalid_argument("Number of colors must be between 2 and 256");
    }

    // Convert image to LAB color space for better quantization
    auto lab_image = rgbToLab(image, width, height);
    
    // Initialize k-means clustering
    std::vector<std::vector<double>> centroids(num_colors, std::vector<double>(3));
    std::vector<int> assignments(width * height);
    
    // Randomly initialize centroids
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < num_colors; i++) {
        int idx = static_cast<int>(dis(gen) * width * height);
        centroids[i][0] = lab_image[idx * 3];
        centroids[i][1] = lab_image[idx * 3 + 1];
        centroids[i][2] = lab_image[idx * 3 + 2];
    }
    
    // K-means clustering
    const int max_iterations = 100;
    bool changed;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        changed = false;
        
        // Assign pixels to nearest centroid
        for (int i = 0; i < width * height; i++) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = 0;
            
            for (int j = 0; j < num_colors; j++) {
                double dist = 0;
                for (int k = 0; k < 3; k++) {
                    double diff = lab_image[i * 3 + k] - centroids[j][k];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            
            if (assignments[i] != best_cluster) {
                assignments[i] = best_cluster;
                changed = true;
            }
        }
        
        if (!changed) break;
        
        // Update centroids
        std::vector<std::vector<double>> new_centroids(num_colors, std::vector<double>(3, 0.0));
        std::vector<int> counts(num_colors, 0);
        
        for (int i = 0; i < width * height; i++) {
            int cluster = assignments[i];
            for (int k = 0; k < 3; k++) {
                new_centroids[cluster][k] += lab_image[i * 3 + k];
            }
            counts[cluster]++;
        }
        
        for (int i = 0; i < num_colors; i++) {
            if (counts[i] > 0) {
                for (int k = 0; k < 3; k++) {
                    centroids[i][k] = new_centroids[i][k] / counts[i];
                }
            }
        }
    }
    
    // Create quantized image
    std::vector<double> quantized(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        int cluster = assignments[i];
        for (int k = 0; k < 3; k++) {
            quantized[i * 3 + k] = centroids[cluster][k];
        }
    }
    
    // Convert back to RGB
    return labToRGB(quantized, width, height);
}

bool ColorSpaces::validateImageDimensions(const std::vector<double>& image,
                                        int expected_width,
                                        int expected_height) {
    return !image.empty() && 
           image.size() == static_cast<size_t>(expected_width * expected_height);
}

std::vector<double> ColorSpaces::computeColorMeans(const std::vector<double>& image,
                                                 int width,
                                                 int height) {
    std::vector<double> means(3, 0.0);
    
    for (int i = 0; i < width * height; i++) {
        for (int k = 0; k < 3; k++) {
            means[k] += image[i * 3 + k];
        }
    }
    
    for (int k = 0; k < 3; k++) {
        means[k] /= (width * height);
    }
    
    return means;
}

std::vector<double> ColorSpaces::computeColorCovariance(const std::vector<double>& image,
                                                      int width,
                                                      int height,
                                                      const std::vector<double>& means) {
    std::vector<double> covariance(9, 0.0);  // 3x3 matrix stored as vector
    
    for (int i = 0; i < width * height; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                double diff_j = image[i * 3 + j] - means[j];
                double diff_k = image[i * 3 + k] - means[k];
                covariance[j * 3 + k] += diff_j * diff_k;
            }
        }
    }
    
    for (int i = 0; i < 9; i++) {
        covariance[i] /= (width * height);
    }
    
    return covariance;
}

} // namespace preprocessing 