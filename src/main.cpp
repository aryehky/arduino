#include <iostream>
#include <vector>
#include "svm.h"
#include "dataset.h"

int main() {
    // Initialize dataset paths (adjust these paths as per your project structure)
    std::string train_images_file = "data/mnist/train-images.idx3-ubyte";
    std::string train_labels_file = "data/mnist/train-labels.idx1-ubyte";
    std::string test_images_file = "data/mnist/t10k-images.idx3-ubyte";
    std::string test_labels_file = "data/mnist/t10k-labels.idx1-ubyte";

    // Create Dataset instance and load data
    Dataset dataset(train_images_file, train_labels_file, test_images_file, test_labels_file);
    dataset.loadTrainingData();
    dataset.loadTestData();

    // Get training and test features and labels
    std::vector<std::vector<double>>& training_features = dataset.getTrainingFeatures();
    std::vector<int>& training_labels = dataset.getTrainingLabels();
    std::vector<std::vector<double>>& test_features = dataset.getTestFeatures();
    std::vector<int>& test_labels = dataset.getTestLabels();

    // Example usage of SVM class (assuming SVM class is implemented in svm.cpp and svm.h)
    SVM svm;
    svm.train(training_features, training_labels);

    // Evaluate model on test dataset
    int correct_predictions = 0;
    for (size_t i = 0; i < test_features.size(); ++i) {
        int predicted_label = svm.predict(test_features[i]);
        if (predicted_label == test_labels[i]) {
            correct_predictions++;
        }
    }

    // Calculate accuracy
    double accuracy = (static_cast<double>(correct_predictions) / test_features.size()) * 100.0;

    // Print results
    std::cout << "SVM model accuracy on test dataset: " << accuracy << "%" << std::endl;

    return 0;
}
