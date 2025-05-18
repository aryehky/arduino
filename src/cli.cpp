#include "cli.h"
#include "svm.h"
#include "dataset.h"
#include "utils.h"
#include <iostream>
#include <stdexcept>

CLI::CLI(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.substr(0, 2) == "--") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                options[arg.substr(2)] = argv[++i];
            } else {
                flags.push_back(arg.substr(2));
            }
        }
    }
}

bool CLI::hasFlag(const std::string& flag) const {
    return std::find(flags.begin(), flags.end(), flag) != flags.end();
}

std::string CLI::getOption(const std::string& option) const {
    auto it = options.find(option);
    return it != options.end() ? it->second : "";
}

void CLI::validateOptions() const {
    if (flags.empty() && options.empty()) {
        showHelp();
        throw std::runtime_error("No options provided");
    }

    if (hasFlag("train") && hasFlag("predict")) {
        throw std::runtime_error("Cannot train and predict simultaneously");
    }
}

void CLI::showHelp() const {
    std::cout << "Usage: digit_recognition [options]\n\n"
              << "Options:\n"
              << "  --train              Train a new model\n"
              << "  --grid-search        Perform grid search during training\n"
              << "  --augment            Use data augmentation during training\n"
              << "  --predict            Run in prediction mode\n"
              << "  --predict-batch DIR  Run batch prediction on directory\n"
              << "  --evaluate           Evaluate model performance\n"
              << "  --visualize          Generate visualizations\n"
              << "  --model PATH         Specify model path\n"
              << "  --help              Show this help message\n";
}

void CLI::handleTraining() {
    std::cout << "Loading dataset...\n";
    Dataset dataset("data/mnist/train-images.idx3-ubyte",
                   "data/mnist/train-labels.idx1-ubyte",
                   "data/mnist/t10k-images.idx3-ubyte",
                   "data/mnist/t10k-labels.idx1-ubyte");
    
    dataset.loadTrainingData();
    dataset.loadTestData();

    auto& training_features = dataset.getTrainingFeatures();
    auto& training_labels = dataset.getTrainingLabels();
    
    SVM svm;

    if (hasFlag("grid-search")) {
        std::cout << "Performing grid search...\n";
        utils::ParamGrid param_grid;
        auto best_params = svm.gridSearch(training_features, training_labels, param_grid);
        
        std::cout << "Best parameters:\n"
                  << "Kernel: " << best_params.best_kernel << "\n"
                  << "C: " << best_params.best_C << "\n"
                  << "Gamma: " << best_params.best_gamma << "\n"
                  << "Validation accuracy: " << best_params.best_accuracy * 100 << "%\n";
    }

    if (hasFlag("augment")) {
        std::cout << "Augmenting training data...\n";
        svm.augmentTrainingData(training_features, training_labels);
    }

    std::cout << "Training model...\n";
    svm.train(training_features, training_labels);

    std::string model_path = getOption("model");
    if (model_path.empty()) {
        model_path = "digit_recognition_model.model";
    }
    
    std::cout << "Saving model to " << model_path << "\n";
    svm.saveModel(model_path);
}

void CLI::handlePrediction() {
    std::string model_path = getOption("model");
    if (model_path.empty()) {
        model_path = "digit_recognition_model.model";
    }

    SVM svm;
    std::cout << "Loading model from " << model_path << "\n";
    svm.loadModel(model_path);

    std::cout << "Enter digit image data (28x28 pixels, space-separated values):\n";
    std::vector<double> features;
    double pixel;
    while (features.size() < 784 && std::cin >> pixel) {
        features.push_back(pixel / 255.0);
    }

    if (features.size() == 784) {
        int prediction = svm.predict(features);
        std::cout << "Predicted digit: " << prediction << "\n";
    } else {
        throw std::runtime_error("Invalid input size");
    }
}

void CLI::handleBatchPrediction() {
    std::string dir_path = getOption("predict-batch");
    if (dir_path.empty()) {
        throw std::runtime_error("No directory specified for batch prediction");
    }

    std::string model_path = getOption("model");
    if (model_path.empty()) {
        model_path = "digit_recognition_model.model";
    }

    SVM svm;
    std::cout << "Loading model from " << model_path << "\n";
    svm.loadModel(model_path);

    // Implementation for batch prediction would go here
    // This would involve reading all images from the specified directory
    std::cout << "Batch prediction not yet implemented\n";
}

void CLI::handleEvaluation() {
    std::string model_path = getOption("model");
    if (model_path.empty()) {
        model_path = "digit_recognition_model.model";
    }

    Dataset dataset("data/mnist/train-images.idx3-ubyte",
                   "data/mnist/train-labels.idx1-ubyte",
                   "data/mnist/t10k-images.idx3-ubyte",
                   "data/mnist/t10k-labels.idx1-ubyte");
    
    dataset.loadTestData();
    auto& test_features = dataset.getTestFeatures();
    auto& test_labels = dataset.getTestLabels();

    SVM svm;
    svm.loadModel(model_path);

    auto confusion_matrix = svm.getConfusionMatrix(test_features, test_labels, 10);
    
    std::cout << "\nEvaluation Results:\n"
              << "Accuracy: " << confusion_matrix.getAccuracy() * 100 << "%\n"
              << "F1 Score: " << confusion_matrix.getF1Score() * 100 << "%\n\n"
              << confusion_matrix.toString() << "\n";
}

void CLI::handleVisualization() {
    // This would be implemented if we add visualization features
    std::cout << "Visualization features not yet implemented\n";
}

void CLI::run() {
    try {
        validateOptions();

        if (hasFlag("help")) {
            showHelp();
            return;
        }

        if (hasFlag("train")) {
            handleTraining();
        } else if (hasFlag("predict")) {
            handlePrediction();
        } else if (hasFlag("predict-batch")) {
            handleBatchPrediction();
        } else if (hasFlag("evaluate")) {
            handleEvaluation();
        } else if (hasFlag("visualize")) {
            handleVisualization();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        throw;
    }
} 