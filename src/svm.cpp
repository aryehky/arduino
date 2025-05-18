#include "svm.h"
#include <svm.h> // LIBSVM header
#include <stdexcept>
#include <cstring>

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
                const std::vector<int>& labels) {
    if (features.empty() || labels.empty() || features.size() != labels.size()) {
        throw std::invalid_argument("Invalid training data");
    }

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
    }

    // Train model
    model->model = svm_train(&model->prob, &model->param);

    // Cleanup
    for (int i = 0; i < model->prob.l; ++i) {
        delete[] model->prob.x[i];
    }
    delete[] model->prob.x;
    delete[] model->prob.y;
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

std::vector<int> SVM::predict(const std::vector<std::vector<double>>& features) {
    std::vector<int> predictions;
    predictions.reserve(features.size());
    for (const auto& feature : features) {
        predictions.push_back(predict(feature));
    }
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