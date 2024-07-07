#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>

class Dataset {
public:
    Dataset(std::string train_images_file, std::string train_labels_file,
            std::string test_images_file, std::string test_labels_file);

    void loadTrainingData();
    void loadTestData();

    std::vector<std::vector<double>>& getTrainingFeatures();
    std::vector<int>& getTrainingLabels();
    std::vector<std::vector<double>>& getTestFeatures();
    std::vector<int>& getTestLabels();

private:
    std::string train_images_file_;
    std::string train_labels_file_;
    std::string test_images_file_;
    std::string test_labels_file_;

    std::vector<std::vector<double>> training_features_;
    std::vector<int> training_labels_;
    std::vector<std::vector<double>> test_features_;
    std::vector<int> test_labels_;

    // Helper functions for data loading and preprocessing
    void readMNISTImages(std::string filename, std::vector<std::vector<double>>& images);
    void readMNISTLabels(std::string filename, std::vector<int>& labels);
};

#endif // DATASET_H
