#include "dataset.h"
#include <fstream>
#include <iostream>
#include <algorithm>

Dataset::Dataset(const std::string& train_images_file,
                const std::string& train_labels_file,
                const std::string& test_images_file,
                const std::string& test_labels_file)
    : train_images_path(train_images_file),
      train_labels_path(train_labels_file),
      test_images_path(test_images_file),
      test_labels_path(test_labels_file) {}

void Dataset::loadTrainingData() {
    readMNISTImages(train_images_path, training_features);
    readMNISTLabels(train_labels_path, training_labels);
    normalizeData();
}

void Dataset::loadTestData() {
    readMNISTImages(test_images_path, test_features);
    readMNISTLabels(test_labels_path, test_labels);
    normalizeData();
}

void Dataset::normalizeData() {
    // Normalize pixel values to [0,1] range
    for (auto& feature_vector : training_features) {
        for (auto& pixel : feature_vector) {
            pixel /= 255.0;
        }
    }
    
    for (auto& feature_vector : test_features) {
        for (auto& pixel : feature_vector) {
            pixel /= 255.0;
        }
    }
}

void Dataset::readMNISTImages(const std::string& filename, std::vector<std::vector<double>>& features) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));

    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    number_of_images = __builtin_bswap32(number_of_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    features.resize(number_of_images);
    int image_size = rows * cols;

    for (int i = 0; i < number_of_images; ++i) {
        features[i].resize(image_size);
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            features[i][j] = static_cast<double>(pixel);
        }
    }
}

void Dataset::readMNISTLabels(const std::string& filename, std::vector<int>& labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int magic_number = 0;
    int number_of_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_labels, sizeof(number_of_labels));

    // Convert from big-endian to little-endian
    magic_number = __builtin_bswap32(magic_number);
    number_of_labels = __builtin_bswap32(number_of_labels);

    labels.resize(number_of_labels);
    for (int i = 0; i < number_of_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i] = static_cast<int>(label);
    }
} 