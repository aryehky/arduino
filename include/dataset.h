#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>

class Dataset {
public:
    Dataset(const std::string& train_images_file,
            const std::string& train_labels_file,
            const std::string& test_images_file,
            const std::string& test_labels_file);

    void loadTrainingData();
    void loadTestData();
    void normalizeData();

    std::vector<std::vector<double>>& getTrainingFeatures() { return training_features; }
    std::vector<int>& getTrainingLabels() { return training_labels; }
    std::vector<std::vector<double>>& getTestFeatures() { return test_features; }
    std::vector<int>& getTestLabels() { return test_labels; }

private:
    std::string train_images_path;
    std::string train_labels_path;
    std::string test_images_path;
    std::string test_labels_path;

    std::vector<std::vector<double>> training_features;
    std::vector<int> training_labels;
    std::vector<std::vector<double>> test_features;
    std::vector<int> test_labels;

    void readMNISTImages(const std::string& filename, std::vector<std::vector<double>>& features);
    void readMNISTLabels(const std::string& filename, std::vector<int>& labels);
};

#endif // DATASET_H
