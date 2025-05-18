#ifndef SVM_H
#define SVM_H

#include <vector>
#include <memory>

struct SVMModel; // Forward declaration for the internal model

class SVM {
public:
    SVM();  // Constructor
    ~SVM(); // Destructor
    
    // Training method
    void train(const std::vector<std::vector<double>>& features, 
               const std::vector<int>& labels);

    // Prediction methods
    int predict(const std::vector<double>& feature);
    std::vector<int> predict(const std::vector<std::vector<double>>& features);

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
    std::vector<double> getPrecisionByClass(const std::vector<std::vector<double>>& test_features,
                                          const std::vector<int>& test_labels);
    std::vector<double> getRecallByClass(const std::vector<std::vector<double>>& test_features,
                                       const std::vector<int>& test_labels);

private:
    std::unique_ptr<SVMModel> model;
    void preprocessFeatures(std::vector<std::vector<double>>& features);
};

#endif // SVM_H
