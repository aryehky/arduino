#ifndef SVM_H
#define SVM_H

#include <vector>
#include <memory>
#include <string>
#include "utils.h"

struct SVMModel; // Forward declaration for the internal model

class SVM {
public:
    SVM();  // Constructor
    ~SVM(); // Destructor
    
    // Training methods
    void train(const std::vector<std::vector<double>>& features, 
               const std::vector<int>& labels,
               const utils::ProgressCallback& progress_callback = nullptr);

    // Cross-validation
    double crossValidate(const std::vector<std::vector<double>>& features,
                        const std::vector<int>& labels,
                        size_t k = 5,
                        const utils::ProgressCallback& progress_callback = nullptr);

    // Grid search
    utils::GridSearchResult gridSearch(const std::vector<std::vector<double>>& features,
                                     const std::vector<int>& labels,
                                     const utils::ParamGrid& param_grid,
                                     size_t k = 5,
                                     const utils::ProgressCallback& progress_callback = nullptr);

    // Prediction methods
    int predict(const std::vector<double>& feature);
    std::vector<int> predict(const std::vector<std::vector<double>>& features,
                           const utils::ProgressCallback& progress_callback = nullptr);

    // Model persistence
    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);

    // Hyperparameter settings
    void setKernelType(const std::string& kernel_type);
    void setC(double C);
    void setGamma(double gamma);

    // Performance metrics
    double getAccuracy(const std::vector<std::vector<double>>& test_features,
                      const std::vector<int>& test_labels);
    utils::ConfusionMatrix getConfusionMatrix(const std::vector<std::vector<double>>& test_features,
                                            const std::vector<int>& test_labels,
                                            size_t num_classes = 10);

    // Data augmentation
    void augmentTrainingData(const std::vector<std::vector<double>>& original_features,
                           const std::vector<int>& original_labels,
                           size_t num_augmented_per_sample = 3,
                           const utils::ProgressCallback& progress_callback = nullptr);

private:
    std::unique_ptr<SVMModel> model;
    void preprocessFeatures(std::vector<std::vector<double>>& features);
    
    // Helper methods for grid search and cross-validation
    double evaluateParams(const std::vector<std::vector<double>>& features,
                         const std::vector<int>& labels,
                         const std::string& kernel_type,
                         double C,
                         double gamma,
                         size_t k);
};

#endif // SVM_H
