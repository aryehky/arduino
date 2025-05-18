#include <iostream>
#include <vector>
#include <iomanip>
#include "svm.h"
#include "dataset.h"
#include "utils.h"

// Progress callback function
void showProgress(float progress, const std::string& status) {
    static int last_percent = -1;
    int percent = static_cast<int>(progress * 100);
    
    if (percent != last_percent) {
        std::cout << "\r" << status << " [";
        int pos = 50 * progress;
        for (int i = 0; i < 50; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "%" << std::flush;
        last_percent = percent;
    }
    
    if (progress >= 1.0) {
        std::cout << std::endl;
    }
}

int main() {
    try {
        // Initialize dataset paths
        std::string train_images_file = "data/mnist/train-images.idx3-ubyte";
        std::string train_labels_file = "data/mnist/train-labels.idx1-ubyte";
        std::string test_images_file = "data/mnist/t10k-images.idx3-ubyte";
        std::string test_labels_file = "data/mnist/t10k-labels.idx1-ubyte";

        std::cout << "Loading MNIST dataset..." << std::endl;

        // Create Dataset instance and load data
        Dataset dataset(train_images_file, train_labels_file, test_images_file, test_labels_file);
        dataset.loadTrainingData();
        dataset.loadTestData();

        // Get training and test data
        auto& training_features = dataset.getTrainingFeatures();
        auto& training_labels = dataset.getTrainingLabels();
        auto& test_features = dataset.getTestFeatures();
        auto& test_labels = dataset.getTestLabels();

        // Create and configure SVM
        SVM svm;

        // Perform grid search to find best parameters
        std::cout << "\nPerforming grid search for optimal parameters..." << std::endl;
        utils::ParamGrid param_grid;
        auto best_params = svm.gridSearch(training_features, training_labels, param_grid, 3, showProgress);

        std::cout << "\nBest parameters found:" << std::endl;
        std::cout << "Kernel: " << best_params.best_kernel << std::endl;
        std::cout << "C: " << best_params.best_C << std::endl;
        std::cout << "Gamma: " << best_params.best_gamma << std::endl;
        std::cout << "Validation accuracy: " << best_params.best_accuracy * 100 << "%" << std::endl;

        // Augment training data
        std::cout << "\nAugmenting training data..." << std::endl;
        svm.augmentTrainingData(training_features, training_labels, 2, showProgress);

        // Train the model with the best parameters
        std::cout << "\nTraining final model..." << std::endl;
        svm.train(training_features, training_labels, showProgress);

        // Evaluate on test set
        std::cout << "\nEvaluating model on test set..." << std::endl;
        auto confusion_matrix = svm.getConfusionMatrix(test_features, test_labels, 10);

        // Print results
        std::cout << "\nTest Results:" << std::endl;
        std::cout << "Accuracy: " << confusion_matrix.getAccuracy() * 100 << "%" << std::endl;
        std::cout << "F1 Score: " << confusion_matrix.getF1Score() * 100 << "%" << std::endl;
        
        std::cout << "\nPrecision by class:" << std::endl;
        auto precision = confusion_matrix.getPrecision();
        for (size_t i = 0; i < precision.size(); ++i) {
            std::cout << "Class " << i << ": " << precision[i] * 100 << "%" << std::endl;
        }

        std::cout << "\nRecall by class:" << std::endl;
        auto recall = confusion_matrix.getRecall();
        for (size_t i = 0; i < recall.size(); ++i) {
            std::cout << "Class " << i << ": " << recall[i] * 100 << "%" << std::endl;
        }

        std::cout << "\nConfusion Matrix:" << std::endl;
        std::cout << confusion_matrix.toString() << std::endl;

        // Save the model
        std::cout << "Saving model..." << std::endl;
        svm.saveModel("digit_recognition_model.model");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
