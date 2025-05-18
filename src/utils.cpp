#include "utils.h"
#include <cmath>
#include <numeric>

namespace utils {

std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> 
createKFolds(size_t dataset_size, size_t k, bool shuffle) {
    std::vector<size_t> indices(dataset_size);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> folds;
    size_t fold_size = dataset_size / k;
    
    for (size_t i = 0; i < k; ++i) {
        std::vector<size_t> test_indices(
            indices.begin() + i * fold_size,
            i == k - 1 ? indices.end() : indices.begin() + (i + 1) * fold_size
        );
        
        std::vector<size_t> train_indices;
        train_indices.reserve(dataset_size - test_indices.size());
        
        for (size_t j = 0; j < dataset_size; ++j) {
            if (std::find(test_indices.begin(), test_indices.end(), indices[j]) == test_indices.end()) {
                train_indices.push_back(indices[j]);
            }
        }
        
        folds.emplace_back(std::move(train_indices), std::move(test_indices));
    }
    
    return folds;
}

ConfusionMatrix::ConfusionMatrix(size_t num_classes) 
    : num_classes(num_classes), matrix(num_classes, std::vector<size_t>(num_classes, 0)) {}

void ConfusionMatrix::update(int true_label, int predicted_label) {
    if (true_label >= 0 && true_label < num_classes && 
        predicted_label >= 0 && predicted_label < num_classes) {
        matrix[true_label][predicted_label]++;
    }
}

std::string ConfusionMatrix::toString() const {
    std::stringstream ss;
    ss << "Confusion Matrix:\n";
    ss << std::setw(8) << " ";
    
    for (size_t i = 0; i < num_classes; ++i) {
        ss << std::setw(6) << i;
    }
    ss << "\n";
    
    for (size_t i = 0; i < num_classes; ++i) {
        ss << std::setw(6) << i << ": ";
        for (size_t j = 0; j < num_classes; ++j) {
            ss << std::setw(6) << matrix[i][j];
        }
        ss << "\n";
    }
    
    return ss.str();
}

double ConfusionMatrix::getAccuracy() const {
    size_t correct = 0;
    size_t total = 0;
    
    for (size_t i = 0; i < num_classes; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            if (i == j) correct += matrix[i][j];
            total += matrix[i][j];
        }
    }
    
    return total > 0 ? static_cast<double>(correct) / total : 0.0;
}

std::vector<double> ConfusionMatrix::getPrecision() const {
    std::vector<double> precision(num_classes);
    
    for (size_t i = 0; i < num_classes; ++i) {
        size_t col_sum = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            col_sum += matrix[j][i];
        }
        precision[i] = col_sum > 0 ? static_cast<double>(matrix[i][i]) / col_sum : 0.0;
    }
    
    return precision;
}

std::vector<double> ConfusionMatrix::getRecall() const {
    std::vector<double> recall(num_classes);
    
    for (size_t i = 0; i < num_classes; ++i) {
        size_t row_sum = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            row_sum += matrix[i][j];
        }
        recall[i] = row_sum > 0 ? static_cast<double>(matrix[i][i]) / row_sum : 0.0;
    }
    
    return recall;
}

double ConfusionMatrix::getF1Score() const {
    auto precision = getPrecision();
    auto recall = getRecall();
    double f1_sum = 0.0;
    
    for (size_t i = 0; i < num_classes; ++i) {
        if (precision[i] + recall[i] > 0) {
            f1_sum += 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]);
        }
    }
    
    return f1_sum / num_classes;
}

std::vector<double> addGaussianNoise(const std::vector<double>& image, double mean, double stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(mean, stddev);
    
    std::vector<double> noisy_image = image;
    for (auto& pixel : noisy_image) {
        pixel = std::max(0.0, std::min(1.0, pixel + d(gen)));
    }
    
    return noisy_image;
}

std::vector<double> rotate(const std::vector<double>& image, int width, int height, double angle) {
    std::vector<double> rotated_image(width * height, 0.0);
    double radian = angle * M_PI / 180.0;
    double cos_theta = std::cos(radian);
    double sin_theta = std::sin(radian);
    
    double x_center = width / 2.0;
    double y_center = height / 2.0;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double xr = cos_theta * (x - x_center) - sin_theta * (y - y_center) + x_center;
            double yr = sin_theta * (x - x_center) + cos_theta * (y - y_center) + y_center;
            
            if (xr >= 0 && xr < width && yr >= 0 && yr < height) {
                int x1 = static_cast<int>(xr);
                int y1 = static_cast<int>(yr);
                int x2 = x1 + 1;
                int y2 = y1 + 1;
                
                if (x2 < width && y2 < height) {
                    double dx = xr - x1;
                    double dy = yr - y1;
                    
                    rotated_image[y * width + x] = 
                        image[y1 * width + x1] * (1 - dx) * (1 - dy) +
                        image[y1 * width + x2] * dx * (1 - dy) +
                        image[y2 * width + x1] * (1 - dx) * dy +
                        image[y2 * width + x2] * dx * dy;
                }
            }
        }
    }
    
    return rotated_image;
}

std::vector<double> translate(const std::vector<double>& image, int width, int height, int dx, int dy) {
    std::vector<double> translated_image(width * height, 0.0);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int new_x = x + dx;
            int new_y = y + dy;
            
            if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                translated_image[new_y * width + new_x] = image[y * width + x];
            }
        }
    }
    
    return translated_image;
}

void updateProgress(float progress, const std::string& status, const ProgressCallback& callback) {
    if (callback) {
        callback(progress, status);
    }
}
} 