#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace utils {

// Progress callback type
using ProgressCallback = std::function<void(float, const std::string&)>;

// Cross-validation utilities
std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> 
createKFolds(size_t dataset_size, size_t k, bool shuffle = true);

// Confusion matrix
class ConfusionMatrix {
public:
    ConfusionMatrix(size_t num_classes);
    void update(int true_label, int predicted_label);
    std::string toString() const;
    double getAccuracy() const;
    std::vector<double> getPrecision() const;
    std::vector<double> getRecall() const;
    double getF1Score() const;

private:
    std::vector<std::vector<size_t>> matrix;
    size_t num_classes;
};

// Data augmentation
std::vector<double> addGaussianNoise(const std::vector<double>& image, double mean = 0.0, double stddev = 0.1);
std::vector<double> rotate(const std::vector<double>& image, int width, int height, double angle);
std::vector<double> translate(const std::vector<double>& image, int width, int height, int dx, int dy);

// Parameter grid search
struct ParamGrid {
    std::vector<std::string> kernel_types = {"linear", "rbf", "polynomial", "sigmoid"};
    std::vector<double> C_values = {0.1, 1.0, 10.0, 100.0};
    std::vector<double> gamma_values = {0.001, 0.01, 0.1, 1.0};
};

struct GridSearchResult {
    std::string best_kernel;
    double best_C;
    double best_gamma;
    double best_accuracy;
};

// Progress reporting
void updateProgress(float progress, const std::string& status, const ProgressCallback& callback);

} // namespace utils

#endif // UTILS_H 