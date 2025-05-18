#include "svm.h"
#include <svm.h> // LIBSVM header
#include <stdexcept>
#include <cstring>
#include <random>
#include <algorithm>

struct SVMModel {
    svm_model* model;
    svm_parameter param;
    svm_problem prob;
};

SVM::SVM() : model(new SVMModel()) {
    // Initialize default parameters
    svm_set_print_string_function([](const char*){});  // Disable console output
    
    std::memset(&model->param, 0, sizeof(model->param));
    model->param.svm_type = C_SVC;
    model->param.kernel_type = RBF;
    model->param.gamma = 0.5;
    model->param.C = 1.0;
    model->param.cache_size = 100;
    model->param.eps = 1e-3;
    model->param.nr_weight = 0;
    model->param.weight_label = nullptr;
    model->param.weight = nullptr;
    model->model = nullptr;
}

SVM::~SVM() {
    if (model) {
        svm_free_and_destroy_model(&model->model);
        svm_destroy_param(&model->param);
        delete model;
    }
}

void SVM::train(const std::vector<std::vector<double>>& features, 
                const std::vector<int>& labels,
                const utils::ProgressCallback& progress_callback) {
    if (features.empty() || labels.empty() || features.size() != labels.size()) {
        throw std::invalid_argument("Invalid training data");
    }

    utils::updateProgress(0.0f, "Preparing training data...", progress_callback);

    // Prepare problem
    model->prob.l = features.size();
    model->prob.y = new double[model->prob.l];
    model->prob.x = new svm_node*[model->prob.l];

    // Convert features to LIBSVM format
    for (int i = 0; i < model->prob.l; ++i) {
        model->prob.y[i] = labels[i];
        model->prob.x[i] = new svm_node[features[i].size() + 1];
        
        for (size_t j = 0; j < features[i].size(); ++j) {
            model->prob.x[i][j].index = j + 1;
            model->prob.x[i][j].value = features[i][j];
        }
        model->prob.x[i][features[i].size()].index = -1;

        if (progress_callback) {
            float progress = static_cast<float>(i) / model->prob.l * 0.5f;
            utils::updateProgress(progress, "Converting features...", progress_callback);
        }
    }

    utils::updateProgress(0.5f, "Training model...", progress_callback);
    
    // Train model
    model->model = svm_train(&model->prob, &model->param);

    // Cleanup
    for (int i = 0; i < model->prob.l; ++i) {
        delete[] model->prob.x[i];
    }
    delete[] model->prob.x;
    delete[] model->prob.y;

    utils::updateProgress(1.0f, "Training completed", progress_callback);
}

int SVM::predict(const std::vector<double>& feature) {
    if (!model->model) {
        throw std::runtime_error("Model not trained");
    }

    // Convert feature to LIBSVM format
    std::vector<svm_node> x(feature.size() + 1);
    for (size_t i = 0; i < feature.size(); ++i) {
        x[i].index = i + 1;
        x[i].value = feature[i];
    }
    x[feature.size()].index = -1;

    double prediction = svm_predict(model->model, x.data());
    return static_cast<int>(prediction);
}

std::vector<int> SVM::predict(const std::vector<std::vector<double>>& features,
                            const utils::ProgressCallback& progress_callback) {
    std::vector<int> predictions;
    predictions.reserve(features.size());

    for (size_t i = 0; i < features.size(); ++i) {
        predictions.push_back(predict(features[i]));

        if (progress_callback) {
            float progress = static_cast<float>(i) / features.size();
            utils::updateProgress(progress, "Making predictions...", progress_callback);
        }
    }

    utils::updateProgress(1.0f, "Predictions completed", progress_callback);
    return predictions;
}

void SVM::setKernelType(const std::string& kernel_type) {
    if (kernel_type == "linear") {
        model->param.kernel_type = LINEAR;
    } else if (kernel_type == "polynomial") {
        model->param.kernel_type = POLY;
    } else if (kernel_type == "rbf") {
        model->param.kernel_type = RBF;
    } else if (kernel_type == "sigmoid") {
        model->param.kernel_type = SIGMOID;
    } else {
        throw std::invalid_argument("Invalid kernel type");
    }
}

void SVM::setC(double C) {
    if (C <= 0) {
        throw std::invalid_argument("C must be positive");
    }
    model->param.C = C;
}

void SVM::setGamma(double gamma) {
    if (gamma <= 0) {
        throw std::invalid_argument("gamma must be positive");
    }
    model->param.gamma = gamma;
}

double SVM::getAccuracy(const std::vector<std::vector<double>>& test_features,
                       const std::vector<int>& test_labels) {
    if (test_features.size() != test_labels.size()) {
        throw std::invalid_argument("Mismatched test data");
    }

    int correct = 0;
    for (size_t i = 0; i < test_features.size(); ++i) {
        if (predict(test_features[i]) == test_labels[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / test_features.size();
}

void SVM::saveModel(const std::string& filename) const {
    if (!model->model) {
        throw std::runtime_error("No model to save");
    }
    if (svm_save_model(filename.c_str(), model->model) != 0) {
        throw std::runtime_error("Failed to save model");
    }
}

void SVM::loadModel(const std::string& filename) {
    if (model->model) {
        svm_free_and_destroy_model(&model->model);
    }
    model->model = svm_load_model(filename.c_str());
    if (!model->model) {
        throw std::runtime_error("Failed to load model");
    }
}

double SVM::crossValidate(const std::vector<std::vector<double>>& features,
                         const std::vector<int>& labels,
                         size_t k,
                         const utils::ProgressCallback& progress_callback) {
    auto folds = utils::createKFolds(features.size(), k);
    double total_accuracy = 0.0;

    for (size_t fold = 0; fold < k; ++fold) {
        utils::updateProgress(static_cast<float>(fold) / k, 
                            "Processing fold " + std::to_string(fold + 1) + "/" + std::to_string(k),
                            progress_callback);

        // Prepare training and validation sets
        std::vector<std::vector<double>> train_features, val_features;
        std::vector<int> train_labels, val_labels;

        for (size_t idx : folds[fold].first) {
            train_features.push_back(features[idx]);
            train_labels.push_back(labels[idx]);
        }

        for (size_t idx : folds[fold].second) {
            val_features.push_back(features[idx]);
            val_labels.push_back(labels[idx]);
        }

        // Train on this fold
        train(train_features, train_labels);

        // Evaluate on validation set
        total_accuracy += getAccuracy(val_features, val_labels);
    }

    utils::updateProgress(1.0f, "Cross-validation completed", progress_callback);
    return total_accuracy / k;
}

utils::GridSearchResult SVM::gridSearch(const std::vector<std::vector<double>>& features,
                                      const std::vector<int>& labels,
                                      const utils::ParamGrid& param_grid,
                                      size_t k,
                                      const utils::ProgressCallback& progress_callback) {
    utils::GridSearchResult best_result;
    best_result.best_accuracy = 0.0;

    size_t total_combinations = param_grid.kernel_types.size() * 
                              param_grid.C_values.size() * 
                              param_grid.gamma_values.size();
    size_t current_combination = 0;

    for (const auto& kernel : param_grid.kernel_types) {
        for (double C : param_grid.C_values) {
            for (double gamma : param_grid.gamma_values) {
                float progress = static_cast<float>(current_combination) / total_combinations;
                std::string status = "Testing kernel=" + kernel + ", C=" + std::to_string(C) + 
                                   ", gamma=" + std::to_string(gamma);
                utils::updateProgress(progress, status, progress_callback);

                double accuracy = evaluateParams(features, labels, kernel, C, gamma, k);

                if (accuracy > best_result.best_accuracy) {
                    best_result.best_accuracy = accuracy;
                    best_result.best_kernel = kernel;
                    best_result.best_C = C;
                    best_result.best_gamma = gamma;
                }

                current_combination++;
            }
        }
    }

    // Set the best parameters
    setKernelType(best_result.best_kernel);
    setC(best_result.best_C);
    setGamma(best_result.best_gamma);

    utils::updateProgress(1.0f, "Grid search completed", progress_callback);
    return best_result;
}

double SVM::evaluateParams(const std::vector<std::vector<double>>& features,
                          const std::vector<int>& labels,
                          const std::string& kernel_type,
                          double C,
                          double gamma,
                          size_t k) {
    // Save current parameters
    auto current_kernel = model->param.kernel_type;
    auto current_C = model->param.C;
    auto current_gamma = model->param.gamma;

    // Set new parameters
    setKernelType(kernel_type);
    setC(C);
    setGamma(gamma);

    // Perform cross-validation
    double accuracy = crossValidate(features, labels, k);

    // Restore original parameters
    model->param.kernel_type = current_kernel;
    model->param.C = current_C;
    model->param.gamma = current_gamma;

    return accuracy;
}

utils::ConfusionMatrix SVM::getConfusionMatrix(const std::vector<std::vector<double>>& test_features,
                                             const std::vector<int>& test_labels,
                                             size_t num_classes) {
    utils::ConfusionMatrix cm(num_classes);
    auto predictions = predict(test_features);

    for (size_t i = 0; i < predictions.size(); ++i) {
        cm.update(test_labels[i], predictions[i]);
    }

    return cm;
}

void SVM::augmentTrainingData(const std::vector<std::vector<double>>& original_features,
                            const std::vector<int>& original_labels,
                            size_t num_augmented_per_sample,
                            const utils::ProgressCallback& progress_callback) {
    std::vector<std::vector<double>> augmented_features;
    std::vector<int> augmented_labels;
    
    size_t total_samples = original_features.size() * num_augmented_per_sample;
    size_t current_sample = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> angle_dist(-15.0, 15.0);
    std::uniform_int_distribution<> shift_dist(-2, 2);

    for (size_t i = 0; i < original_features.size(); ++i) {
        for (size_t j = 0; j < num_augmented_per_sample; ++j) {
            // Apply random augmentations
            std::vector<double> augmented = original_features[i];
            
            if (j % 3 == 0) {
                augmented = utils::addGaussianNoise(augmented);
            } else if (j % 3 == 1) {
                augmented = utils::rotate(augmented, 28, 28, angle_dist(gen));
            } else {
                augmented = utils::translate(augmented, 28, 28, 
                                          shift_dist(gen), shift_dist(gen));
            }

            augmented_features.push_back(augmented);
            augmented_labels.push_back(original_labels[i]);

            if (progress_callback) {
                float progress = static_cast<float>(current_sample) / total_samples;
                utils::updateProgress(progress, "Generating augmented samples...", progress_callback);
                current_sample++;
            }
        }
    }

    // Add augmented data to training set
    train(augmented_features, augmented_labels, progress_callback);
} 